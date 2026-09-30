"""Readers for the files scripts/eval/generate.py writes, shared by the gate
and every judge so all of them see exactly the same responses.

Layout under <run_dir>/eval/ (tag = <adapter_tag>_finaldata_temp1.0_seed0):
  OEQ/PAS    responses_<tag>[_shard{i}of{N}].csv          id, sentence, response, ...
  AITA/FLIP  aita_responses_<tag>_shard{i}of{N}.csv        see generate.py
  gate loop  gate/responses_<tag>_gateloop.csv / gate/aita_responses_<tag>_gateloop.csv

A "key" is (id, side): side is "" except for AITA-NTA-FLIP, whose two
responses per pair are "original" and "flipped".
"""

import glob
import os
import re

import pandas as pd

from plurpo.data import AITA_ENVS

# FLIP: (side, context column, response column, gate_passed column)
FLIP_SIDES = (("original", "post_text", "response_original", "gate_passed_original"),
              ("flipped", "flipped_post_text", "response_flipped", "gate_passed_flipped"))


def _stem(env):
    return "aita_responses" if env in AITA_ENVS else "responses"


def resolve_shards(run_dir, env, tag, expected_shards=None):
    """Files of the one complete attempt-1 shard set for `tag`.

    Groups files by their declared shard count N (an unsuffixed OEQ/PAS file
    counts as N=1) and returns the complete group -- exactly `expected_shards`
    when given, else the largest complete N. A run dir can hold an abandoned
    partial regime next to a complete one; concatenating both would duplicate
    rows, and an incomplete set would score over the wrong denominator, so both
    are fatal."""
    eval_dir = os.path.join(run_dir, "eval")
    groups = {}
    for f in sorted(glob.glob(os.path.join(eval_dir, f"{_stem(env)}_{tag}_shard*.csv"))):
        m = re.search(r"_shard(\d+)of(\d+)\.csv$", f)
        assert m, f"unparsable shard filename: {f}"
        groups.setdefault(int(m.group(2)), []).append(f)
    single = os.path.join(eval_dir, f"{_stem(env)}_{tag}.csv")
    if os.path.exists(single):
        groups.setdefault(1, []).append(single)
    assert groups, f"no attempt-1 responses matching {eval_dir}/{_stem(env)}_{tag}*.csv"
    if expected_shards is not None:
        chosen = expected_shards
        assert chosen in groups, f"{tag}: no of{chosen} shard set; have {sorted(groups)}"
    else:
        complete = [n for n, fs in groups.items() if len(fs) == n]
        assert complete, f"{tag}: no complete shard set { {n: len(f) for n, f in groups.items()} }"
        chosen = max(complete)
    assert len(groups[chosen]) == chosen, f"{tag}: of{chosen} has {len(groups[chosen])} files"
    return sorted(groups[chosen], key=lambda f: int(re.search(r"_shard(\d+)of", f).group(1))
                  if "_shard" in f else 0)


def load_attempt1(run_dir, env, tag, expected_shards=None):
    """Concatenated attempt-1 frame (shard order), ids asserted unique."""
    files = resolve_shards(run_dir, env, tag, expected_shards)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    assert df["id"].is_unique, f"{tag}: duplicate ids across shards"
    df["id"] = df["id"].astype(str)
    return df


def load_gateloop(run_dir, env, tag):
    """The gate-loop frame for `tag`, or None if no key failed attempt 1."""
    p = os.path.join(run_dir, "eval", "gate", f"{_stem(env)}_{tag}_gateloop.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["id"] = df["id"].astype(str)
    return df


def keyed(df, env, only_passed=False):
    """Flatten a responses frame to one row per key:
    (id, side, context, response). With only_passed, keep keys whose
    gate_passed[_side] is true (used on gate-loop frames). FLIP rows are
    emitted all-original then all-flipped."""
    if env == "AITA-NTA-FLIP":
        parts = []
        for side, ctx, resp, gcol in FLIP_SIDES:
            part = df[df[gcol].astype(bool)] if only_passed else df
            part = part[["id", ctx, resp]].copy()
            part.columns = ["id", "context", "response"]
            part["side"] = side
            parts.append(part)
        out = pd.concat(parts, ignore_index=True)
    else:
        part = df[df["gate_passed"].astype(bool)] if only_passed else df
        ctx = "post_text" if env in AITA_ENVS else "sentence"
        out = part[["id", ctx, "response"]].copy()
        out.columns = ["id", "context", "response"]
        out["side"] = ""
    out["id"] = out["id"].astype(str)
    return out[["id", "side", "context", "response"]].reset_index(drop=True)
