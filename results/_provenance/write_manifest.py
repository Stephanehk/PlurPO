"""Write results/MANIFEST.csv: path, size, sha256 and origin of every shipped
file (generated results/_reproduced/ is excluded). Run from anywhere."""

import csv
import hashlib
import os

import pandas as pd

RES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COPIES = {"veto_vs_endorse/": "<research>/_veto_endorse_probe_qwen_oeq/"}


def origins():
    src = {}
    d = pd.read_csv(os.path.join(RES, "arms.csv"), keep_default_na=False)
    for _, r in d.iterrows():
        src[r["file"]] = (f"built from {r['source_run']}/eval ({r['gate_source']} gate, "
                          f"tag {r['source_tag']})" if r["source_run"] else "built from OEQ human responses")
    return src


def main():
    src = origins()
    rows = []
    for root, _dirs, files in os.walk(RES):
        for fn in files:
            p = os.path.relpath(os.path.join(root, fn), RES)
            if p.startswith("_reproduced") or "__pycache__" in p or p == "MANIFEST.csv":
                continue
            origin = src.get(p, "")
            for k, v in COPIES.items():
                if p.startswith(k):
                    origin = "copy of " + v
            if p.startswith("_provenance") or p in ("arms.csv", "README.md"):
                origin = "written for this release"
            with open(os.path.join(RES, p), "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            rows.append((p, os.path.getsize(os.path.join(RES, p)), h, origin))
    rows.sort()
    with open(os.path.join(RES, "MANIFEST.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "bytes", "sha256", "origin"])
        w.writerows(rows)
    print(f"{len(rows)} files -> MANIFEST.csv")


if __name__ == "__main__":
    main()
