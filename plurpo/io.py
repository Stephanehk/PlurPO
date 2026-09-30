"""Small file helpers shared by the training and evaluation scripts.

JSONL files in this project are written one flushed, newline-terminated record
at a time, so a job killed mid-write can only leave a truncated FINAL line.
`read_jsonl` drops such a trailing fragment and parses everything else; an
unparsable interior line is real corruption and raises.

Paths ending in `.gz` are read / written gzip-compressed (the large released
pair files in data/ are stored that way to stay under GitHub's file limit).
"""

import csv
import gzip
import json
import os

csv.field_size_limit(10**9)


def _open_text(path, mode):
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode)


def read_jsonl(path):
    """Return the list of records in `path` (see module docstring)."""
    with _open_text(path, "r") as f:
        text = f.read()
    lines = text.split("\n")
    trailing = lines.pop()
    if trailing.strip():
        print(f"[io] dropping truncated final line ({len(trailing)} chars) of "
              f"{os.path.basename(path)}")
    return [json.loads(line) for line in lines if line.strip()]


def write_jsonl(path, records):
    """Write `records` to `path`, one JSON object per line."""
    with _open_text(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_keyed_candidates(path):
    """candidates.jsonl -> {(id, sample_idx): row}, last write wins (a resumed
    sampler may rewrite a key); dict order is first-occurrence order."""
    rows = {}
    for r in read_jsonl(path):
        rows[(str(r["id"]), int(r["sample_idx"]))] = r
    return rows


def read_labels_csv(path):
    """labels.csv -> {(id, sample_idx): int label}."""
    labels = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            labels[(str(r["id"]), int(r["sample_idx"]))] = int(r["label"])
    return labels
