"""Build the normalized per-arm result frames in results/ from the original
research artifacts (provenance; runs only on the research cluster).

For every (model, env, arm) in registry.ARMS this reconstructs the FINAL
response set after rejection sampling -- one row per held-out prompt -- and
attaches every judge label the paper uses, reproducing the join logic of the
original table builders (build_gated_finaldata_tables.py,
gated_{endorse,verdict}_table_32b.py, build_5judge_gated_tables.py):

  * a key that passed the engagement gate at attempt 1 keeps its attempt-1
    response and its stored labels;
  * a key rescued by a later draw takes the rescued response and the labels
    judged on it;
  * a key no draw rescued within 5 attempts gets gate_passed = 0 and no labels.

Every join asserts that prompt (and, where available, response) text lines up,
so a mis-keyed label aborts the build instead of silently mislabelling an arm.

Output: results/<model>/<ENV>/<arm>.csv and results/arms.csv.
Run:    python results/_provenance/build_results.py
"""

import glob
import json
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from registry import ARMS, RUNS, V5, V5G, V1, JD, TAGSUF, require_root  # noqa: E402
from elephant_join import attach_elephant  # noqa: E402

OUT = os.path.dirname(HERE)
MAX_ATTEMPTS = 5
csv_kw = dict(keep_default_na=False, na_values=[""])


def _norm(s):
    return s.astype(str).str.strip()


def read_shards(run, prefix, tag):
    """Concatenate <prefix><tag><TAGSUF>_shard{i}of{N}.csv in shard order,
    asserting shards 0..N-1 are all present."""
    d = f"{RUNS}/{run}/eval"
    hits = glob.glob(f"{d}/{prefix}{tag}{TAGSUF}_shard*of*.csv")
    assert hits, f"{run}: no {prefix}{tag} shards"
    ns = {int(re.search(r"of(\d+)\.csv$", h).group(1)) for h in hits}
    assert len(ns) == 1, f"{run}/{tag}: mixed shard regimes {ns}"
    n = ns.pop()
    paths = [f"{d}/{prefix}{tag}{TAGSUF}_shard{i}of{n}.csv" for i in range(n)]
    assert all(os.path.exists(p) for p in paths), f"{run}/{tag}: missing shards"
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["id"] = df["id"].astype(str)
    assert df["id"].is_unique, f"{run}/{tag}: duplicate ids"
    return df


def gate_a1(run, tag):
    g = pd.read_csv(f"{RUNS}/{run}/eval/gate/gate_{tag}{TAGSUF}_a1.csv")
    g["id"] = g["id"].astype(str)
    g["side"] = g["side"].fillna("").astype(str)
    return g


def _text_label_map(path, col="endorse_response"):
    """(sentence, response) -> label for a scored file; asserts unambiguous."""
    d = pd.read_csv(path)
    out = {}
    for s, r, lab in zip(_norm(d["sentence"]), _norm(d["response"]), d[col]):
        k = (s, r)
        assert k not in out or out[k] == lab, f"{path}: conflicting duplicate {k[0][:60]!r}"
        out[k] = lab
    return out


# ------------------------------------------------------------------ OEQ/PAS
def endorse_inloop(env, run, tag, x):
    """Gated OEQ/PAS frame for an arm whose gate ran inside generation."""
    sh = read_shards(run, "responses_", tag)
    fz = pd.read_csv(f"{V5}/{x['v5']}")
    assert len(fz) == len(sh), f"{x['v5']}: {len(fz)} rows != {len(sh)} generated"
    assert (_norm(fz["sentence"]) == _norm(sh["sentence"])).all(), f"{x['v5']}: join"
    frame = pd.DataFrame({"id": sh["id"], "sentence": sh["sentence"],
                          "response": sh["response"]})
    frame["endorse_response"] = fz["endorse_response"].astype("Int64").values
    frame["gate_passed"] = 1
    frame["n_tried"] = 1
    a1 = gate_a1(run, tag)
    passers = set(a1[a1.gate == 1]["id"])
    failed = ~frame["id"].isin(passers)
    glp = f"{RUNS}/{run}/eval/gate/responses_{tag}{TAGSUF}_gateloop.csv"
    if not failed.any():
        return frame
    assert os.path.exists(glp), f"{run}/{tag}: attempt-1 failures but no gateloop file"
    gl = pd.read_csv(glp)
    gl["id"] = gl["id"].astype(str)
    passed = gl[gl.gate_passed == 1].reset_index(drop=True)
    labs = []
    if len(passed):
        sc = pd.read_csv(f"{V5G}/scored_{x['v5g']}_gateloop.csv")
        assert len(sc) == len(passed), f"{x['v5g']}: {len(sc)} != {len(passed)} rescued"
        assert (_norm(sc["sentence"]) == _norm(passed["sentence"])).all(), f"{x['v5g']}: join"
        labs = sc["endorse_response"].tolist()
    lab_of = dict(zip(passed["id"], labs))
    gl = gl.set_index("id")
    for i in frame.index[failed]:
        k = frame.at[i, "id"]
        assert k in gl.index, f"{run}/{tag}: failed key {k} missing from gateloop"
        frame.at[i, "n_tried"] = int(gl.at[k, "n_tried"])
        if int(gl.at[k, "gate_passed"]) == 1:
            frame.at[i, "response"] = gl.at[k, "response"]
            frame.at[i, "endorse_response"] = int(lab_of[k])
        else:
            frame.at[i, "gate_passed"] = 0
            frame.at[i, "response"] = None
            frame.at[i, "endorse_response"] = pd.NA
    return frame


def endorse_staged(env, run, tag, x):
    """Gated OEQ/PAS frame for an arm whose gate ran as separate rounds."""
    sh = read_shards(run, "responses_", tag)
    gf = pd.read_csv(f"{RUNS}/{run}/eval/gate/gatefinal_{tag}{TAGSUF}.csv")
    gf["id"] = gf["id"].astype(str)
    frozen = _text_label_map(f"{V5}/{x['v5']}")
    rp = f"{V5G}/scored_{x['v5g']}_gateloop.csv"
    rescued = _text_label_map(rp) if os.path.exists(rp) else {}
    gf = gf.set_index("id")
    rows = []
    for _, r in sh.iterrows():
        k = r["id"]
        if k not in gf.index:
            rows.append((k, r["sentence"], None, pd.NA, 0, MAX_ATTEMPTS))
            continue
        g = gf.loc[k]
        wa = int(g["winning_attempt"])
        src = frozen if wa == 1 else rescued
        key = (str(g["sentence"]).strip(), str(g["response"]).strip())
        assert key in src, f"{run}/{tag}: no label for key {k} (attempt {wa})"
        rows.append((k, g["sentence"], g["response"], int(src[key]), 1, wa))
    frame = pd.DataFrame(rows, columns=["id", "sentence", "response",
                                        "endorse_response", "gate_passed", "n_tried"])
    st = json.load(open(f"{RUNS}/{run}/eval/gate/gatestats_{tag}{TAGSUF}.json"))
    assert int((frame.gate_passed == 0).sum()) == int(st["n_failed"]), f"{run}/{tag}: n_failed"
    return frame


def endorse_human(x):
    """OEQ human responses (never gated), keyed to the final test split ids."""
    sys.path.insert(0, os.path.dirname(OUT))
    from plurpo.data import load_split
    test = load_split("OEQ", "test")
    fz = pd.read_csv(f"{V5}/{x['v5']}")
    assert len(fz) == len(test)
    id_of = {s: i for s, i in zip(_norm(test["post_text"]), test["id"])}
    assert len(id_of) == len(test), "duplicate OEQ test prompts"
    ids = [id_of[s] for s in _norm(fz["sentence"])]
    assert len(set(ids)) == len(ids)
    return pd.DataFrame({"id": ids, "sentence": fz["sentence"], "response": fz["response"],
                         "endorse_response": fz["endorse_response"].astype("Int64"),
                         "gate_passed": 1, "n_tried": pd.NA})


def attach_v1(frame, x):
    """Original-prompt (v1) endorsement label, joined on (sentence, response);
    the v1 files were judged on the already-gated responses."""
    if not x.get("v1") or not os.path.exists(f"{V1}/{x['v1']}"):
        return frame
    m = _text_label_map(f"{V1}/{x['v1']}")
    g = frame["gate_passed"] == 1
    labs = []
    for s, r, ok in zip(_norm(frame["sentence"]), frame["response"], g):
        labs.append(m[(s, str(r).strip())] if ok else pd.NA)
    assert len(m) == int(g.sum()) or len(m) >= int(g.sum()), "v1 coverage"
    frame["endorse_v1_response"] = pd.array(labs, dtype="Int64")
    return frame


# ------------------------------------------------------------------ AITA / FLIP
def verdict_frame(env, run, tag, kind):
    """Gated AITA / AITA-NTA-FLIP frame (attempt-1 verdicts, rescued verdicts
    swapped in, exhausted sides marked gate_passed = 0)."""
    df = read_shards(run, "aita_scored_", tag)
    a1 = gate_a1(run, tag)
    rp = f"{RUNS}/{run}/eval/gate/aita_scored_{tag}_gateloop.csv"
    resc = None
    if os.path.exists(rp):
        resc = pd.read_csv(rp)
        resc["id"] = resc["id"].astype(str)
        resc = resc.set_index("id")
    gf = None
    gfp = f"{RUNS}/{run}/eval/gate/gatefinal_{tag}{TAGSUF}.csv"
    if kind == "staged":
        gf = pd.read_csv(gfp)
        gf["id"] = gf["id"].astype(str)
        gf = gf.set_index("id")
    if env == "AITA":
        sides = [("", "", "pred", "response")]
        out = df[["id", "post_text", "response", "is_asshole", "pred"]].copy()
    else:
        sides = [("original", "_original", "pred_original", "response_original"),
                 ("flipped", "_flipped", "pred_flipped", "response_flipped")]
        out = df[["id", "post_text", "flipped_post_text", "response_original",
                  "response_flipped", "pred_original", "pred_flipped"]].copy()
    for side, suf, pred_col, resp_col in sides:
        gp_col, nt_col = f"gate_passed{suf}", f"n_tried{suf}"
        out[gp_col] = 1
        out[nt_col] = 1
        pass1 = set(a1[(a1.gate == 1) & (a1.side == side)]["id"])
        for i, r in out.iterrows():
            k = r["id"]
            if k in pass1:
                continue
            if resc is not None and k in resc.index and int(resc.at[k, gp_col]) == 1:
                out.at[i, pred_col] = resc.at[k, pred_col]
                out.at[i, resp_col] = resc.at[k, resp_col]
                if nt_col in resc.columns:
                    out.at[i, nt_col] = int(resc.at[k, nt_col])
                else:
                    out.at[i, nt_col] = int(resc.at[k, f"winning_attempt{suf}"])
            else:
                out.at[i, gp_col] = 0
                out.at[i, nt_col] = MAX_ATTEMPTS
                out.at[i, pred_col] = None
                out.at[i, resp_col] = None
        if gf is not None:
            # staged: per-side draw counts come from gatefinal's winning attempt
            wa = gf[f"winning_attempt{suf}"]
            for i, r in out.iterrows():
                if r[gp_col] == 1:
                    assert r["id"] in gf.index, f"{run}/{tag}: {r['id']} not in gatefinal"
                    out.at[i, nt_col] = int(wa.loc[r["id"]])
    if env == "AITA" and "is_asshole" in out:
        out["is_asshole"] = out["is_asshole"].astype(int)
    return out


# ------------------------------------------------------------------ main
def build(model, env, arm, kind, run, tag, x):
    if env in ("OEQ", "PAS"):
        if kind == "human":
            frame = endorse_human(x)
        elif kind == "inloop":
            frame = endorse_inloop(env, run, tag, x)
        else:
            frame = endorse_staged(env, run, tag, x)
        frame = attach_v1(frame, x)
    else:
        frame = verdict_frame(env, run, tag, kind)
    if x.get("jslug"):
        frame = attach_elephant(frame, env, arm, run, tag, x)
    return frame


PROMPT_COLS = ("sentence", "post_text", "flipped_post_text")


def main():
    require_root()
    index = []
    for model, env, arm, kind, run, tag, x in ARMS:
        frame = build(model, env, arm, kind, run, tag, x)
        rel = f"{model}/{env}/{arm}.csv"
        os.makedirs(os.path.dirname(f"{OUT}/{rel}"), exist_ok=True)
        # prompt text is not shipped (it is in data/splits/, joined on id)
        frame = frame.drop(columns=[c for c in PROMPT_COLS if c in frame.columns])
        frame.to_csv(f"{OUT}/{rel}", index=False)
        index.append({"model": model, "env": env, "arm": arm, "file": rel,
                      "gate_source": kind, "source_run": run or "",
                      "source_tag": (tag + TAGSUF) if tag else ""})
        print(f"[build] {rel}: {len(frame)} rows")
    pd.DataFrame(index).to_csv(f"{OUT}/arms.csv", index=False)


if __name__ == "__main__":
    main()
