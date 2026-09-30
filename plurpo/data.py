"""Dataset loading for the four ELEPHANT environments.

The frozen train/eval splits live in `data/splits/` as
`<stem>_<split>.csv` (+ the authoritative `<stem>_<split>_ids.json`). These CSVs
are byte-identical copies of the rows the original research code selected from
the raw ELEPHANT / OSF files, so no raw source file is needed to train or
evaluate.

Environment names match the on-disk run names used throughout the paper's
artifacts: "OEQ", "PAS", "AITA", "AITA-NTA-FLIP" (the paper's AITA-Flipped).

Assumptions (asserted):
  - every split CSV has a string `id` column whose order equals its ids json;
  - OEQ/PAS/AITA rows are single prompts (`post_text`); AITA-NTA-FLIP rows are
    (original, flipped) PAIRS (`post_text`, `flipped_post_text`).

The model always sees the BARE post as its user message, for every env (no
output-format nudge; AITA verdicts are read from the prose by an eval-time
classifier).
"""

import json
import os

import pandas as pd

from plurpo.paths import SPLITS_DIR, TUNING_DIR

ENVS = ("OEQ", "PAS", "AITA", "AITA-NTA-FLIP")
STEM = {"OEQ": "oeq", "PAS": "pas", "AITA": "aita", "AITA-NTA-FLIP": "flip"}
AITA_ENVS = ("AITA", "AITA-NTA-FLIP")

# AITA-NTA-FLIP carries two prompts per row. They are keyed "<id>::orig" and
# "<id>::flip" wherever per-prompt ids are needed, so a response to the
# original post can never be pooled with one to the flipped post.
FLIP_VARIANTS = (("post_text", "orig"), ("flipped_post_text", "flip"))


def _split_dir(split_set):
    """Directory for a named split set: "final" (the paper's train/eval data)
    or "tuning" (the held-out OEQ/PAS data used only to tune PlurPO)."""
    assert split_set in ("final", "tuning"), f"unknown split set {split_set!r}"
    return SPLITS_DIR if split_set == "final" else TUNING_DIR


def load_ids(env, split, split_set="final"):
    """Ordered id list for one split (the authoritative split artifact)."""
    assert env in STEM, f"unknown env {env!r}"
    assert split in ("train", "test"), f"split must be train/test, got {split!r}"
    path = os.path.join(_split_dir(split_set), f"{STEM[env]}_{split}_ids.json")
    assert os.path.exists(path), f"missing split ids file: {path}"
    with open(path) as f:
        ids = json.load(f)
    assert isinstance(ids, list) and ids, f"empty/malformed ids file: {path}"
    return [str(i) for i in ids]


def load_split(env, split, split_set="final"):
    """Load one split as a DataFrame, rows in ids-file order.

    Columns: OEQ `id, post_text, human_response`; PAS `id, post_text`;
    AITA `id, title, body, verdict, is_asshole, post_text`;
    AITA-NTA-FLIP `id, post_text, flipped_post_text`.
    `keep_default_na=False` keeps literal strings such as "NA" intact.
    """
    path = os.path.join(_split_dir(split_set), f"{STEM[env]}_{split}.csv")
    assert os.path.exists(path), f"missing split csv: {path}"
    df = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
    assert list(df["id"]) == load_ids(env, split, split_set), (
        f"{path}: row order does not match its ids json")
    assert "post_text" in df.columns, f"{path}: no post_text column"
    if env == "AITA-NTA-FLIP":
        assert "flipped_post_text" in df.columns, f"{path}: no flipped_post_text"
    return df


def train_prompts(env, df):
    """Flatten a split DataFrame into the ordered list of bare training prompts.

    One prompt per row for OEQ/PAS/AITA. For AITA-NTA-FLIP each pair expands to
    TWO prompts, original then flipped, fed independently through PlurPO. The
    order is deterministic given the row order.
    """
    assert env in ENVS, f"unknown env {env!r}"
    if env != "AITA-NTA-FLIP":
        return df["post_text"].tolist()
    prompts = []
    for orig, flip in zip(df["post_text"].tolist(), df["flipped_post_text"].tolist()):
        prompts.append(orig)
        prompts.append(flip)
    return prompts


def keyed_prompts(env, df):
    """-> [(prompt_id, post_text), ...] for a split DataFrame.

    OEQ/PAS/AITA use the row id. AITA-NTA-FLIP emits both variants per row,
    keyed "<id>::orig" / "<id>::flip" (see FLIP_VARIANTS). Asserts the keys
    are unique.
    """
    if env != "AITA-NTA-FLIP":
        out = [(str(i), t) for i, t in zip(df["id"], df["post_text"])]
    else:
        out = [(f"{r['id']}::{suffix}", r[col])
               for _, r in df.iterrows()
               for col, suffix in FLIP_VARIANTS]
    keys = [k for k, _ in out]
    assert len(set(keys)) == len(keys), f"{env}: duplicate prompt ids"
    return out
