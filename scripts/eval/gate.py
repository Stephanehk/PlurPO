"""Rejection-sampling gate, attempt 1: judge every held-out response and write
the worklist of prompts to resample.

The gate (gpt-5-mini, prompt in plurpo/prompts/judges.py: GATE_PROMPT) labels
each response 1 if it engages the user's situation (an assistant answering,
however well or badly, whether it agrees or disagrees) and 0 otherwise
(critiquing the message as a text, continuing it, role-play/new post,
degenerate output, assessing it as a narrative). It is deliberately orthogonal
to the endorsement axis.

Pipeline for one arm (see slurm/run_eval_all.sh):
  generate.py                  attempt-1 responses (GPU)
  gate.py                      THIS: attempt-1 labels + worklist (network)
  generate.py --gate-loop      resample the worklist inline, attempts 2..5
                               (GPU + network)
  judge_endorsement.py / classify_verdicts.py
                               score attempt-1 passers and rescued prompts;
                               prompts never rescued are dropped and tallied.

Keys are (id, side): side is "" except for AITA-NTA-FLIP, whose original and
flipped responses are gated separately ("original" / "flipped").

Outputs under <run-dir>/eval/gate/:
  gate_<tag>_a1.csv           id, side, attempt, gate, raw, ptok, ctok
  resample_<tag>_a2.json      {"env","tag","attempt","n_keys","n_ids","keys","ids"}
                              only when at least one key was rejected.
Idempotent: an existing gate_<tag>_a1.csv is reused (worklist rewritten).
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from plurpo.data import ENVS
from plurpo.eval_outputs import keyed, load_attempt1
from plurpo.judges import gate_one
from plurpo.llm_api import SpendGuard, make_client


def judge_frame(df, env, workers, max_cost):
    """Gate-label every row; every row must get a parsed 0/1 (asserted)."""
    client = make_client()
    guard = SpendGuard(max_cost)

    def work(r):
        label, res = gate_one(client, env, r.context, r.response)
        guard.add(res)
        return label, res["content"].strip(), res["prompt_tokens"], res["completion_tokens"]

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(work, list(df.itertuples(index=False))))
    out = df.copy()
    out["gate"] = [r[0] for r in res]
    out["raw"] = [r[1] for r in res]
    out["ptok"] = [r[2] for r in res]
    out["ctok"] = [r[3] for r in res]
    print(f"[gate] spend ${guard.spent:.4f}")
    return out


def write_worklist(path, env, tag, keys):
    """Persist the rejected keys; `ids` sorted so the resample order depends
    only on the worklist contents."""
    ids = sorted({str(k["id"]) for k in keys})
    payload = {"env": env, "tag": tag, "attempt": 2, "n_keys": len(keys),
               "n_ids": len(ids),
               "keys": sorted(keys, key=lambda k: (str(k["id"]), k.get("side", ""))),
               "ids": ids}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--env", required=True, choices=ENVS)
    ap.add_argument("--tag", required=True,
                    help="e.g. base-perspshiftassess_finaldata_temp1.0_seed0")
    ap.add_argument("--expected-shards", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-cost", type=float, default=None, help="informational USD cap")
    args = ap.parse_args()

    gate_dir = os.path.join(args.run_dir, "eval", "gate")
    os.makedirs(gate_dir, exist_ok=True)
    gate_path = os.path.join(gate_dir, f"gate_{args.tag}_a1.csv")
    if os.path.exists(gate_path):
        print(f"[gate] reusing {gate_path}")
        out = pd.read_csv(gate_path, keep_default_na=False, dtype={"id": str, "side": str})
        out["gate"] = out["gate"].astype(int)
    else:
        df = keyed(load_attempt1(args.run_dir, args.env, args.tag, args.expected_shards),
                   args.env)
        print(f"[gate] env={args.env} tag={args.tag} rows={len(df)}")
        out = judge_frame(df, args.env, args.workers, args.max_cost)
        out["attempt"] = 1
        out[["id", "side", "attempt", "gate", "raw", "ptok", "ctok"]].to_csv(gate_path, index=False)
        print(f"[gate] wrote {gate_path}")

    n_pass = int((out["gate"] == 1).sum())
    print(f"[gate] pass {n_pass}/{len(out)}  reject {len(out) - n_pass}")
    rejected = out[out["gate"] == 0]
    wl_path = os.path.join(gate_dir, f"resample_{args.tag}_a2.json")
    if len(rejected) == 0:
        print("[gate] every key passed at attempt 1; no worklist")
        return
    keys = [{"id": str(r.id), "side": str(r.side)} for r in rejected.itertuples(index=False)]
    payload = write_worklist(wl_path, args.env, args.tag, keys)
    print(f"[gate] worklist: {payload['n_keys']} keys over {payload['n_ids']} ids -> {wl_path}")


if __name__ == "__main__":
    main()
