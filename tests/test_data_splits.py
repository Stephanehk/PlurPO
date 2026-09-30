"""Integrity checks for the released train/eval splits (data/splits, data/tuning).

Checks, for every split: row counts, unique ids, and train/test disjointness by
id and by whitespace/case-normalised prompt text. Two text-level overlaps are
known and expected, and are asserted to be exactly these (so any new overlap
fails):
  - PAS: train pas_01a2c64d is test pas_d92572cf up to whitespace. The source
    ids hash the exact text, so the variant got a different id. Pairs built
    from it are removed from the released pair files (see data/README.md).
  - AITA-NTA-FLIP: the source ELEPHANT file has the literal flipped narration
    "ERROR" for 4 train and 8 test pairs, so those "ERROR" texts coincide.

Run: pytest tests/test_data_splits.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plurpo.data import ENVS, load_split

FINAL_COUNTS = {"OEQ": (1000, 1000), "PAS": (1000, 1000), "AITA": (1000, 1000),
                "AITA-NTA-FLIP": (500, 1000)}
TUNING_COUNTS = {"OEQ": (1000, 1000), "PAS": (1000, 1000)}
EXPECTED_TEXT_OVERLAP = {
    ("final", "PAS"): {("pas_01a2c64d", "pas_d92572cf")},
    ("final", "AITA-NTA-FLIP"): {"error"},
}


def _norm(text):
    return " ".join(str(text).split()).lower()


def _texts(env, df):
    cols = ["post_text", "flipped_post_text"] if env == "AITA-NTA-FLIP" else ["post_text"]
    return {_norm(r[c]): r["id"] for _, r in df.iterrows() for c in cols}


def _check(split_set, counts):
    for env, (n_train, n_test) in counts.items():
        train = load_split(env, "train", split_set)
        test = load_split(env, "test", split_set)
        assert len(train) == n_train and len(test) == n_test, (split_set, env)
        assert train["id"].is_unique and test["id"].is_unique, (split_set, env)
        assert not set(train["id"]) & set(test["id"]), (split_set, env)
        tr, te = _texts(env, train), _texts(env, test)
        shared = set(tr) & set(te)
        expected = EXPECTED_TEXT_OVERLAP.get((split_set, env), set())
        if env == "AITA-NTA-FLIP":
            assert shared == expected, (split_set, env, shared)
        else:
            assert {(tr[k], te[k]) for k in shared} == expected, (split_set, env, shared)


def test_final_splits():
    assert set(FINAL_COUNTS) == set(ENVS)
    _check("final", FINAL_COUNTS)


def test_tuning_splits():
    _check("tuning", TUNING_COUNTS)


def test_aita_balanced():
    for split in ("train", "test"):
        counts = load_split("AITA", split)["is_asshole"].astype(int).value_counts().to_dict()
        assert counts == {0: 500, 1: 500}, (split, counts)


def test_flip_error_rows():
    for split, n in (("train", 4), ("test", 8)):
        df = load_split("AITA-NTA-FLIP", split)
        assert int((df["flipped_post_text"].str.strip().str.lower() == "error").sum()) == n
        assert int((df["post_text"].str.strip().str.lower() == "error").sum()) == 0
