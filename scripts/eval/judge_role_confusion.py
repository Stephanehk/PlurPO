"""Role-confusion (narrative-voice) judge: does a response answer as an
OUTSIDER (an assistant/advisor who knows the events only from the user's
message) or as a STAKEHOLDER inside the user's story (claims to have lived the
events)? gpt-5-mini via OpenRouter, prompt ROLE_CONFUSION_TEMPLATE.

The role-confusion rate = #STAKEHOLDER / #(STAKEHOLDER + OUTSIDER + UNCLEAR),
with a Wilson 95% interval; ERR / PARSE_FAIL judgements are reported and
excluded. Used for the role-confusion table and the RPO-alpha sweep
(appendix). This is an evaluation instrument; the training-time voice filter
of PlurPO's role-confusion stage uses a different rubric
(plurpo/prompts/role_confusion.py).

Input: an arm's attempt-1 responses (default) or its rescued gate-loop
responses (--source gateloop). FLIP judges both sides of every pair.
Output: <run-dir>/eval/voice_<tag>[_gateloop].csv with columns
id, side, voice. Idempotent unless --overwrite.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from plurpo.data import ENVS
from plurpo.eval_outputs import keyed, load_attempt1, load_gateloop
from plurpo.metrics import Z95_EXACT, wilson
from plurpo.judges import role_confusion_one
from plurpo.llm_api import make_client
from plurpo.prompts.judges import ROLE_CONFUSION_LABELS


def report(out, env):
    """Print the STAKEHOLDER rate with its Wilson interval."""
    bad = out[~out["voice"].isin(ROLE_CONFUSION_LABELS)]
    if len(bad):
        print(f"[voice] WARNING {len(bad)} unusable judgements (PARSE_FAIL)")
    ok = out[out["voice"].isin(ROLE_CONFUSION_LABELS)]
    k = int((ok["voice"] == "STAKEHOLDER").sum())
    p, lo, hi = wilson(k, len(ok), z=Z95_EXACT)
    print(f"[voice] {env}: STAKEHOLDER {k}/{len(ok)} = {p:.3f}  Wilson 95% [{lo:.3f}, {hi:.3f}]")
    print(ok["voice"].value_counts().to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--env", required=True, choices=ENVS)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--source", default="attempt1", choices=("attempt1", "gateloop"))
    ap.add_argument("--expected-shards", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    suffix = "_gateloop" if args.source == "gateloop" else ""
    out_path = os.path.join(args.run_dir, "eval", f"voice_{args.tag}{suffix}.csv")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[voice] {out_path} exists; --overwrite to redo")
        report(pd.read_csv(out_path), args.env)
        return
    if args.source == "attempt1":
        items = keyed(load_attempt1(args.run_dir, args.env, args.tag, args.expected_shards),
                      args.env)
    else:
        g = load_gateloop(args.run_dir, args.env, args.tag)
        assert g is not None, f"no gate-loop responses for {args.tag}"
        items = keyed(g, args.env, only_passed=True)
    print(f"[voice] env={args.env} judgements={len(items)}")
    client = make_client()
    rows = list(items.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        voices = list(ex.map(
            lambda r: role_confusion_one(client, args.env, r.context, r.response)[0], rows))
    out = pd.DataFrame({"id": items["id"], "side": items["side"], "voice": voices})
    out.to_csv(out_path, index=False)
    print(f"[voice] wrote {out_path}")
    report(out, args.env)


if __name__ == "__main__":
    main()
