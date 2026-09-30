"""Build results/ frames from a FRESH run of the evaluation pipeline.

Turns the outputs of scripts/eval/{generate,gate,judge_endorsement,
classify_verdicts,judge_elephant,judge_role_confusion}.py for one arm into the
frame the analysis scripts read, results/<model>/<ENV>/<arm>.csv (after
rejection sampling). Prompt text (`sentence`, `post_text`, `flipped_post_text`)
is dropped from the written frame; it is in data/splits/, joined on `id`.

Rejection-sampled frame, per prompt (and per side for AITA-NTA-FLIP):
  * passed the gate at attempt 1  -> attempt-1 response + its labels, n_tried 1
  * failed, rescued by a later draw -> rescued response + labels judged on it,
    n_tried = draws used
  * failed, never rescued          -> gate_passed 0, no response / labels
The attempt-1 gate labels (gate_<tag>_a1.csv) are the authority on which keys
were rejected (the FLIP gate loop resamples both sides of a pair).

Inputs (TAG = <adapter tag>_finaldata_temp1.0_seed0, e.g. base-becritical_...):
  <run>/eval/responses_<TAG>[_shard*].csv          OEQ/PAS generations
  <run>/eval/aita_scored_<TAG>_shard*.csv          AITA/FLIP generations + verdicts
  <run>/eval/gate/gate_<TAG>_a1.csv                attempt-1 gate labels
  <run>/eval/gate/[aita_]responses_<TAG>_gateloop.csv   rescued draws
  <run>/eval/gate/aita_scored_<adapter tag>_gateloop.csv  verdicts on rescued draws
  <scores>/endorse_gpt5mini_v5_temp0/scored_<endorse-name>.csv
  <scores>/endorse_gpt5mini_v5_temp0_gateloop/scored_<endorse-gateloop-name>.csv
  <scores>/endorse_gpt5mini_v1_gated/scored_<v1-name>.csv        (optional)
  <elephant-dir>/{framing,validation}_<elephant-name>.csv          (optional)
  <elephant-dir>/{framing,validation}_<elephant-gateloop-name>.csv (optional)
Scored files are joined on `id` (and `side`) when they carry those columns,
otherwise positionally; either way prompt and response text are asserted to
line up, so a mis-keyed label aborts instead of mislabelling an arm.

Example (one arm):
  python scripts/analysis/build_frames.py --model qwen3-8b --env OEQ --arm plurpo \\
      --run-dir runs/dpo_lora_oeq_... --tag final --scores-root outputs/scores \\
      --endorse-name oeq_plurpo --endorse-gateloop-name oeq_plurpo_gateloop \\
      --elephant-name plurpo --elephant-gateloop-name plurpo_gateloop
"""

import argparse
import glob
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO  # noqa: E402,F401  (puts the repo on sys.path)
from plurpo.data import load_split  # noqa: E402
from plurpo.paths import RESULTS_DIR  # noqa: E402

MAX_ATTEMPTS = 5
TAGSUF = "_finaldata_temp1.0_seed0"
SIDES = {"AITA-NTA-FLIP": (("original", "_original"), ("flipped", "_flipped"))}


def _norm(s):
    return s.astype(str).str.strip()


def read_shards(eval_dir, prefix, full_tag, lexicographic=False):
    """Concatenate <prefix><full_tag>[_shard{i}of{N}].csv in shard order
    (numeric; `lexicographic=True` gives the file-name sort order, e.g.
    shard10 before shard2, which some research-era judge files followed)."""
    single = os.path.join(eval_dir, f"{prefix}{full_tag}.csv")
    hits = glob.glob(os.path.join(eval_dir, f"{prefix}{full_tag}_shard*of*.csv"))
    if not hits:
        assert os.path.exists(single), f"no {prefix}{full_tag} outputs in {eval_dir}"
        df = pd.read_csv(single)
    else:
        ns = {int(re.search(r"of(\d+)\.csv$", h).group(1)) for h in hits}
        assert len(ns) == 1, f"{full_tag}: mixed shard regimes {ns}"
        n = ns.pop()
        paths = [os.path.join(eval_dir, f"{prefix}{full_tag}_shard{i}of{n}.csv") for i in range(n)]
        if lexicographic:
            paths = sorted(paths)
        assert all(os.path.exists(p) for p in paths), f"{full_tag}: missing shards of {n}"
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["id"] = df["id"].astype(str)
    assert df["id"].is_unique, f"{full_tag}: duplicate ids"
    return df


def gate_a1(eval_dir, full_tag):
    g = pd.read_csv(os.path.join(eval_dir, "gate", f"gate_{full_tag}_a1.csv"))
    g["id"] = g["id"].astype(str)
    g["side"] = g["side"].fillna("").astype(str)
    return g


def align(scored, frame, text_cols, what):
    """Return `scored` reordered to `frame`'s rows: by `id` (+ `side`) when the
    scored file has them, else positionally. Asserts the text columns match."""
    if "id" in scored.columns:
        scored = scored.copy()
        scored["id"] = scored["id"].astype(str)
        keys = ["id"] + (["side"] if "side" in scored.columns and "side" in frame.columns else [])
        if "side" in keys:
            scored["side"] = scored["side"].fillna("").astype(str)
        scored = scored.set_index(keys).loc[pd.MultiIndex.from_frame(frame[keys])
                                            if len(keys) > 1 else frame[keys[0]]]
        scored = scored.reset_index()
    assert len(scored) == len(frame), f"{what}: {len(scored)} scored rows != {len(frame)}"
    for sc, fc in text_cols:
        assert (_norm(scored[sc]).values == _norm(frame[fc]).values).all(), \
            f"{what}: {sc} does not line up with the generations"
    return scored.reset_index(drop=True)


# ------------------------------------------------------------------ OEQ / PAS
def endorse_frame(a):
    ev = os.path.join(a.run_dir, "eval")
    full = a.tag + TAGSUF
    sh = read_shards(ev, "responses_", full)
    fz = pd.read_csv(os.path.join(a.scores_root, "endorse_gpt5mini_v5_temp0",
                                  f"scored_{a.endorse_name}.csv"))
    fz = align(fz, sh, [("sentence", "sentence")], a.endorse_name)
    frame = pd.DataFrame({"id": sh["id"], "sentence": sh["sentence"], "response": sh["response"]})
    frame["endorse_response"] = fz["endorse_response"].astype("Int64").values
    frame["gate_passed"] = 1
    frame["n_tried"] = 1
    a1 = gate_a1(ev, full)
    failed = ~frame["id"].isin(set(a1[a1.gate == 1]["id"]))
    if not failed.any():
        return frame
    glp = os.path.join(ev, "gate", f"responses_{full}_gateloop.csv")
    assert os.path.exists(glp), f"attempt-1 failures but no gate-loop file {glp}"
    gl = pd.read_csv(glp)
    gl["id"] = gl["id"].astype(str)
    passed = gl[gl.gate_passed == 1].reset_index(drop=True)
    lab_of = {}
    if len(passed):
        sc = pd.read_csv(os.path.join(a.scores_root, "endorse_gpt5mini_v5_temp0_gateloop",
                                      f"scored_{a.endorse_gateloop_name}.csv"))
        sc = align(sc, passed, [("sentence", "sentence")], a.endorse_gateloop_name)
        lab_of = dict(zip(passed["id"], sc["endorse_response"]))
    gl = gl.set_index("id")
    for i in frame.index[failed]:
        k = frame.at[i, "id"]
        assert k in gl.index, f"failed key {k} missing from the gate-loop file"
        frame.at[i, "n_tried"] = int(gl.at[k, "n_tried"])
        if int(gl.at[k, "gate_passed"]) == 1:
            frame.at[i, "response"] = gl.at[k, "response"]
            frame.at[i, "endorse_response"] = int(lab_of[k])
        else:
            frame.at[i, "gate_passed"] = 0
            frame.at[i, "response"] = None
            frame.at[i, "endorse_response"] = pd.NA
    return frame


def human_frame(a):
    """OEQ human reference (judge_endorsement.py --source human): never gated."""
    fz = pd.read_csv(os.path.join(a.scores_root, "endorse_gpt5mini_v5_temp0",
                                  f"scored_{a.endorse_name}.csv"))
    test = load_split("OEQ", "test")
    if "id" not in fz.columns:
        id_of = dict(zip(_norm(test["post_text"]), test["id"]))
        fz["id"] = [id_of[s] for s in _norm(fz["sentence"])]
    fz["id"] = fz["id"].astype(str)
    assert sorted(fz["id"]) == sorted(test["id"]), "human scores do not cover the test split"
    return pd.DataFrame({"id": fz["id"], "sentence": fz["sentence"], "response": fz["response"],
                         "endorse_response": fz["endorse_response"].astype("Int64"),
                         "gate_passed": 1, "n_tried": pd.NA})


def attach_v1(frame, a):
    """Original-prompt judge labels (judge_endorsement.py --source gated
    --prompt-version v1), joined on (sentence, response) of gated rows."""
    path = os.path.join(a.scores_root, "endorse_gpt5mini_v1_gated", f"scored_{a.v1_name}.csv")
    d = pd.read_csv(path)
    m = {}
    for s, r, lab in zip(_norm(d["sentence"]), _norm(d["response"]), d["endorse_response"]):
        assert (s, r) not in m or m[(s, r)] == lab, f"{path}: conflicting duplicate"
        m[(s, r)] = lab
    labs = [m[(str(s).strip(), str(r).strip())] if ok == 1 else pd.NA
            for s, r, ok in zip(frame["sentence"], frame["response"], frame["gate_passed"])]
    frame["endorse_v1_response"] = pd.array(labs, dtype="Int64")
    return frame


# ------------------------------------------------------------------ AITA / FLIP
def verdict_frame(a):
    ev = os.path.join(a.run_dir, "eval")
    full = a.tag + TAGSUF
    df = read_shards(ev, "aita_scored_", full)
    a1 = gate_a1(ev, full)
    rp = os.path.join(ev, "gate", f"aita_scored_{a.tag}_gateloop.csv")
    resc = None
    if os.path.exists(rp):
        resc = pd.read_csv(rp)
        resc["id"] = resc["id"].astype(str)
        resc = resc.set_index("id")
    if a.env == "AITA":
        sides = [("", "")]
        out = df[["id", "post_text", "response", "is_asshole", "pred"]].copy()
    else:
        sides = SIDES["AITA-NTA-FLIP"]
        out = df[["id", "post_text", "flipped_post_text", "response_original",
                  "response_flipped", "pred_original", "pred_flipped"]].copy()
    for side, suf in sides:
        gp, nt, pc, rc = f"gate_passed{suf}", f"n_tried{suf}", f"pred{suf}", f"response{suf}"
        out[gp] = 1
        out[nt] = 1
        pass1 = set(a1[(a1.gate == 1) & (a1.side == side)]["id"])
        for i, r in out.iterrows():
            k = r["id"]
            if k in pass1:
                continue
            if resc is not None and k in resc.index and int(resc.at[k, gp]) == 1:
                out.at[i, pc] = resc.at[k, pc]
                out.at[i, rc] = resc.at[k, rc]
                out.at[i, nt] = int(resc.at[k, nt])
            else:
                out.at[i, gp] = 0
                out.at[i, nt] = MAX_ATTEMPTS
                out.at[i, pc] = None
                out.at[i, rc] = None
    if a.env == "AITA":
        out["is_asshole"] = out["is_asshole"].astype(int)
    return out


# ------------------------------------------------------------------ ELEPHANT
def _elephant_response_frame(frame, env):
    """(id, side, sentence, response) rows in the order judge_elephant.py
    judged an arm's generations (FLIP: all originals, then all flipped)."""
    if env != "AITA-NTA-FLIP":
        sent = "sentence" if "sentence" in frame.columns else "post_text"
        return pd.DataFrame({"id": frame["id"], "side": "", "sentence": frame[sent],
                             "response": frame["response"]})
    parts = [pd.DataFrame({"id": frame["id"], "side": side, "sentence": frame[post],
                           "response": frame[f"response_{side}"]})
             for side, post in (("original", "post_text"), ("flipped", "flipped_post_text"))]
    return pd.concat(parts, ignore_index=True)


def _lines_up(scored, fr):
    return len(scored) == len(fr) and all(
        (_norm(scored[c]).values == _norm(fr[c]).values).all() for c in ("sentence", "response"))


def attach_elephant(frame, a, attempt1_frame, first_lex):
    """ELEPHANT labels: attempt-1 passers from <metric>_<elephant-name>.csv,
    rescued responses from <metric>_<elephant-gateloop-name>.csv (joined on
    id, side). Asserts every gate-passing row/side got a label."""
    fr = _elephant_response_frame(attempt1_frame, a.env)
    ev = os.path.join(a.run_dir, "eval")
    a1 = gate_a1(ev, a.tag + TAGSUF) if a.run_dir else None
    for metric in ("framing", "validation"):
        col = f"{metric}_score_response"
        d = pd.read_csv(os.path.join(a.elephant_dir, f"{metric}_{a.elephant_name}.csv"))
        fr_used = fr
        if "id" not in d.columns and not _lines_up(d, fr):
            # research-era judge files (no id column) followed the file-name
            # order of the shards; accept exactly that alternative order
            fr_used = _elephant_response_frame(first_lex, a.env)
        d = align(d, fr_used, [("sentence", "sentence"), ("response", "response")],
                  f"{metric}_{a.elephant_name}")
        fr_ids, fr_sides = fr_used["id"], fr_used["side"]
        lab = {(i, s): v for i, s, v in zip(fr_ids, fr_sides, d[col])}
        if a1 is not None:
            passers = set(zip(a1[a1.gate == 1]["id"], a1[a1.gate == 1]["side"]))
            lab = {k: v for k, v in lab.items() if k in passers}
        rescued = {}
        if a.elephant_gateloop_name:
            rp = os.path.join(a.elephant_dir, f"{metric}_{a.elephant_gateloop_name}.csv")
            r = pd.read_csv(rp)
            r["id"] = r["id"].astype(str)
            r["side"] = r["side"].fillna("").astype(str) if "side" in r.columns else ""
            for i, s, v, resp in zip(r["id"], r["side"], r[col], r["response"]):
                if a1 is None or (i, s) not in lab:     # a1 passers keep their label
                    rescued[(i, s)] = (v, str(resp).strip())
        sides = SIDES.get(a.env, (("", ""),))
        for side, suf in sides:
            vals = []
            for _, row in frame.iterrows():
                if row[f"gate_passed{suf}"] != 1:
                    vals.append(pd.NA)
                    continue
                k = (row["id"], side)
                if k in lab:
                    vals.append(lab[k])
                    continue
                v, resp = rescued[k]
                assert resp == str(row[f"response{suf}"]).strip(), f"{metric}: rescued text {k}"
                vals.append(v)
            name = f"{metric}_response" if a.env != "AITA-NTA-FLIP" else f"{metric}_{side}"
            vals = pd.array(vals, dtype="Int64")
            assert pd.Series(vals)[frame[f"gate_passed{suf}"].values == 1].notna().all(), \
                f"{metric}: missing labels"
            frame[name] = vals
    return frame


def attach_human_elephant(frame, a):
    for metric in ("framing", "validation"):
        d = pd.read_csv(os.path.join(a.elephant_dir, f"{metric}_{a.elephant_name}.csv"))
        m = {(s, r): v for s, r, v in zip(_norm(d["sentence"]), _norm(d["response"]),
                                          d[f"{metric}_score_response"])}
        frame[f"{metric}_response"] = pd.array(
            [m[(str(s).strip(), str(r).strip())] for s, r in zip(frame["sentence"], frame["response"])],
            dtype="Int64")
    return frame


def build(a):
    if a.env in ("OEQ", "PAS"):
        if a.arm == "human":
            frame = human_frame(a)
            if a.v1_name:
                frame = attach_v1(frame, a)
            if a.elephant_name:
                frame = attach_human_elephant(frame, a)
            return frame
        frame = endorse_frame(a)
        if a.v1_name:
            frame = attach_v1(frame, a)
        first = read_shards(os.path.join(a.run_dir, "eval"), "responses_", a.tag + TAGSUF)
    else:
        frame = verdict_frame(a)
        first = read_shards(os.path.join(a.run_dir, "eval"), "aita_responses_", a.tag + TAGSUF)
    if a.elephant_name:
        prefix = "responses_" if a.env in ("OEQ", "PAS") else "aita_responses_"
        lex = read_shards(os.path.join(a.run_dir, "eval"), prefix, a.tag + TAGSUF, lexicographic=True)
        frame = attach_elephant(frame, a, first, lex)
    return frame


PROMPT_COLS = ("sentence", "post_text", "flipped_post_text")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="e.g. qwen3-8b")
    ap.add_argument("--env", required=True, choices=["OEQ", "PAS", "AITA", "AITA-NTA-FLIP"])
    ap.add_argument("--arm", required=True, help="arm key, e.g. base, plurpo, human")
    ap.add_argument("--run-dir", default=None, help="generate.py run dir (not for --arm human)")
    ap.add_argument("--tag", default=None, help="adapter tag: final, base, base-becritical, ...")
    ap.add_argument("--scores-root", default=None, help="judge_endorsement.py --scores-root")
    ap.add_argument("--endorse-name", default=None)
    ap.add_argument("--endorse-gateloop-name", default=None)
    ap.add_argument("--v1-name", default=None)
    ap.add_argument("--elephant-dir", default=None,
                    help="dir holding <metric>_<name>.csv (judge_elephant.py writes "
                         "<scores-root>/elephant/<ENV>/)")
    ap.add_argument("--elephant-name", default=None)
    ap.add_argument("--elephant-gateloop-name", default=None)
    ap.add_argument("--out", default=None, help="override the output csv path")
    a = ap.parse_args(argv)
    if a.env in ("OEQ", "PAS"):
        assert a.scores_root and a.endorse_name, "OEQ/PAS need --scores-root and --endorse-name"
    if a.arm != "human":
        assert a.run_dir and a.tag, "--run-dir and --tag are required"
    if a.elephant_name:
        assert a.elephant_dir, "--elephant-name needs --elephant-dir"
    return a


def main(argv=None):
    a = parse_args(argv)
    frame = build(a)
    frame = frame.drop(columns=[c for c in PROMPT_COLS if c in frame.columns])
    out = a.out or os.path.join(RESULTS_DIR, a.model, a.env, f"{a.arm}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"[build_frames] {len(frame)} rows -> {out}")
    return out


if __name__ == "__main__":
    main()
