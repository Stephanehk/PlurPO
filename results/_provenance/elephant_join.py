"""Attach ELEPHANT framing / validation judge labels to a gated arm frame
(provenance; ports build_5judge_gated_tables.gated_binary_labels).

Attempt-1 passers take their label from judges_finaldata/<ENV>/<metric>_<slug>.csv,
which is positionally aligned with the arm's generation shards (FLIP: originals
then flipped). Gate-rescued responses take theirs from
judges_finaldata/_gated_rescued/<ENV>_<metric>_rescued.csv, whose row order is
rebuilt from the gate-loop files exactly as the original judging script built
it. A side counts as rescued only if it FAILED at attempt 1 (on FLIP the gate
loop resampled both sides of a pair; the attempt-1 label is the authority).
Prompt and response text are asserted to match at every join.
"""

import glob
import os

import pandas as pd

from registry import RUNS, JD, TAGSUF

RESC = f"{JD}/_gated_rescued"
JDIR = {"OEQ": "OEQ", "PAS": "PAS", "AITA": "AITA", "AITA-NTA-FLIP": "FLIP"}
METRICS = ("framing", "validation")

# Verbatim from build_5judge_gated_tables.RESCUE: arms whose gate-loop files fed
# the rescued judge frames, in the order they were concatenated.
RESCUE = {
 "OEQ": [("becritical", "baseline_final_OEQ", "base-becritical"),
         ("perspshift", "baseline_final_OEQ", "base-perspshiftassess"),
         ("sft", "sft_lora_oeq_qwen3-8b_becritical_final", "final"),
         ("neutralmixed", "dpo_lora_oeq_qwen3-8b_n1000_neutral_mixed_final", "final"),
         ("voice2", "dpo_lora_oeq_qwen3-8b_P6_voice2_rpo0p5", "final")],
 "PAS": [("base", "baseline_final_PAS", "base"), ("explicit", "baseline_final_PAS", "base-explicit"),
         ("sharma", "baseline_final_PAS", "base-sharma"), ("dontbesyco", "baseline_final_PAS", "base-dontbesyco"),
         ("becritical", "baseline_final_PAS", "base-becritical"), ("perspshift", "baseline_final_PAS", "base-perspshiftassess"),
         ("sft", "sft_lora_pas_qwen3-8b_becritical_final", "final"),
         ("neutralmixed", "dpo_lora_pas_qwen3-8b_n1000_neutral_mixed_final", "final"),
         ("neutral", "dpo_lora_pas_qwen3-8b_n1000_neutral_final", "final"),
         ("voice2", "dpo_lora_pas_qwen3-8b_P6_voice2_rpo0p5", "final")],
 "AITA": [("perspshift", "baseline_final_AITA", "base-perspshiftassess"),
          ("sharma", "baseline_final_AITA", "base-sharma"),
          ("voice2", "dpo_lora_aita_qwen3-8b_P6_voice2R2_rpo0p5", "final")],
 "AITA-NTA-FLIP": [
          ("base", "baseline_final_AITA-NTA-FLIP", "base"), ("sharma", "baseline_final_AITA-NTA-FLIP", "base-sharma"),
          ("dontbesyco", "baseline_final_AITA-NTA-FLIP", "base-dontbesyco"), ("becritical", "baseline_final_AITA-NTA-FLIP", "base-becritical"),
          ("explicit", "baseline_final_AITA-NTA-FLIP", "base-explicit"), ("perspshift", "baseline_final_AITA-NTA-FLIP", "base-perspshiftassess"),
          ("voice2", "dpo_lora_flip_qwen3-8b_P6_voice2R2_rpo0p5", "final")],
}
_KEYS_CACHE = {}


def _norm(s):
    return s.astype(str).str.strip()


def response_frame(env, run, tag):
    """(id, side, sentence, response) of the arm's attempt-1 generations in the
    order the ELEPHANT judges saw them."""
    d = f"{RUNS}/{run}/eval"
    aita = env in ("AITA", "AITA-NTA-FLIP")
    pref = "aita_responses_" if aita else "responses_"
    shards = sorted(glob.glob(f"{d}/{pref}{tag}{TAGSUF}_shard*of*.csv"))
    assert shards, f"{run}/{tag}: no response shards"
    df = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
    df["id"] = df["id"].astype(str)
    if env == "AITA-NTA-FLIP":
        return pd.concat([
            pd.DataFrame({"id": df["id"], "side": "original",
                          "sentence": df["post_text"], "response": df["response_original"]}),
            pd.DataFrame({"id": df["id"], "side": "flipped",
                          "sentence": df["flipped_post_text"], "response": df["response_flipped"]}),
        ], ignore_index=True)
    sent = "post_text" if aita else "sentence"
    return pd.DataFrame({"id": df["id"], "side": "", "sentence": df[sent],
                         "response": df["response"]})


def rescued_keys(env, metric):
    """Stored rescued labels re-keyed to (arm, id, side, is_rescued)."""
    if (env, metric) in _KEYS_CACHE:
        return _KEYS_CACHE[(env, metric)]
    stored = pd.read_csv(f"{RESC}/{JDIR[env]}_{metric}_rescued.csv")
    rows = []
    aita = env in ("AITA", "AITA-NTA-FLIP")
    pref = "aita_responses_" if aita else "responses_"
    for slug, run, tag in RESCUE[env]:
        p = f"{RUNS}/{run}/eval/gate/{pref}{tag}{TAGSUF}_gateloop.csv"
        if not os.path.exists(p):
            continue
        r = pd.read_csv(p)
        r["id"] = r["id"].astype(str)
        g = pd.read_csv(f"{RUNS}/{run}/eval/gate/gate_{tag}{TAGSUF}_a1.csv")
        g["id"] = g["id"].astype(str)
        g["side"] = g["side"].fillna("").astype(str)
        failed = set(zip(g[g.gate == 0]["id"], g[g.gate == 0]["side"]))
        if env == "AITA-NTA-FLIP":
            for _, x in r.iterrows():
                for side, gp, wa, resp in (
                        ("original", "gate_passed_original", "winning_attempt_original", "response_original"),
                        ("flipped", "gate_passed_flipped", "winning_attempt_flipped", "response_flipped")):
                    if x.get(gp, 0) == 1 and int(x[wa]) >= 2:
                        rows.append((slug, x["id"], side, x[resp], (x["id"], side) in failed))
        else:
            for _, x in r[r["gate_passed"] == 1].iterrows():
                rows.append((slug, x["id"], "", x["response"], (x["id"], "") in failed))
    keys = pd.DataFrame(rows, columns=["arm", "id", "side", "response", "is_rescued"])
    assert len(keys) == len(stored), f"{env}/{metric}: {len(keys)} rebuilt != {len(stored)} stored"
    assert (_norm(keys["response"]) == _norm(stored["response"])).all(), f"{env}/{metric}: rescued join"
    assert (keys["arm"].values == stored["arm"].values).all(), f"{env}/{metric}: arm order"
    keys[f"{metric}_response"] = stored[f"{metric}_response"].values
    _KEYS_CACHE[(env, metric)] = keys
    return keys


def _labels_for(env, metric, arm, run, tag, x):
    """{(id, side): label} for the gate-passing responses of one arm."""
    judged = pd.read_csv(f"{JD}/{JDIR[env]}/{metric}_{x['jslug']}.csv")
    lab = pd.to_numeric(judged[f"{metric}_response"], errors="coerce")
    assert lab.notna().all(), f"{env}/{metric}/{x['jslug']}: unparsable labels"
    if run is None:                      # human reference: joined by text
        return {("text", s, r): int(v) for s, r, v in
                zip(_norm(judged["sentence"]), _norm(judged["response"]), lab)}
    fr = response_frame(env, run, tag)
    assert len(fr) == len(judged), f"{env}/{x['jslug']}: {len(fr)} != {len(judged)}"
    assert (_norm(fr["sentence"]) == _norm(judged["sentence"])).all(), f"{env}/{x['jslug']}: prompt join"
    assert (_norm(fr["response"]) == _norm(judged["response"])).all(), f"{env}/{x['jslug']}: response join"
    g = pd.read_csv(f"{RUNS}/{run}/eval/gate/gate_{tag}{TAGSUF}_a1.csv")
    g["id"] = g["id"].astype(str)
    g["side"] = g["side"].fillna("").astype(str)
    passers = set(zip(g[g.gate == 1]["id"], g[g.gate == 1]["side"]))
    out = {}
    for i, s, v in zip(fr["id"], fr["side"], lab):
        if (i, s) in passers:
            out[(i, s)] = (int(v), None)
    keys = rescued_keys(env, metric)
    m = (keys["arm"] == x["rslug"]) & keys["is_rescued"]
    for _, k in keys[m].iterrows():
        out[(k["id"], k["side"])] = (int(k[f"{metric}_response"]), str(k["response"]).strip())
    return out


def attach_elephant(frame, env, arm, run, tag, x):
    """Add <metric>_response (single envs) or <metric>_original/_flipped (FLIP)
    columns; asserts every gate-passing row got a label."""
    for metric in METRICS:
        labs = _labels_for(env, metric, arm, run, tag, x)
        if env == "AITA-NTA-FLIP":
            for side in ("original", "flipped"):
                col = []
                for _, r in frame.iterrows():
                    if r[f"gate_passed_{side}"] != 1:
                        col.append(pd.NA)
                        continue
                    v, resp = labs[(r["id"], side)]
                    assert resp is None or resp == str(r[f"response_{side}"]).strip()
                    col.append(v)
                frame[f"{metric}_{side}"] = pd.array(col, dtype="Int64")
            continue
        col = []
        for _, r in frame.iterrows():
            if r["gate_passed"] != 1:
                col.append(pd.NA)
                continue
            if run is None:
                col.append(labs[("text", str(r["sentence"]).strip(), str(r["response"]).strip())])
                continue
            v, resp = labs[(r["id"], "")]
            assert resp is None or resp == str(r["response"]).strip(), f"{env}/{arm}: rescued text"
            col.append(v)
        frame[f"{metric}_response"] = pd.array(col, dtype="Int64")
    return frame
