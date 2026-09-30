# data/splits — frozen train / evaluation splits

The train and held-out evaluation splits used for every result in the paper
(except the hyper-parameter / prompt tuning, which used `data/tuning/`).
Loaded by `plurpo.data.load_split(env, split)`.

The `*_ids.json` files are the authoritative artifact; the `*.csv` files
materialise the rows, byte-identical to the rows the research code selected
from the raw sources (verified field-by-field when this release was built).

| env (`plurpo.data` name) | stem | train | test | columns |
|---|---|---|---|---|
| OEQ | `oeq` | 1000 | 1000 | `id, post_text, human_response` |
| PAS | `pas` | 1000 | 1000 | `id, post_text` |
| AITA | `aita` | 1000 | 1000 | `id, title, body, verdict, is_asshole, post_text` |
| AITA-NTA-FLIP (AITA-Flipped) | `flip` | 500 pairs | 1000 pairs | `id, post_text, flipped_post_text` |

Read with `dtype={"id": str}, keep_default_na=False` (as `plurpo.data` does).

## How each split was drawn (`MANIFEST.json`, seed 20260710)

Test was sampled first, then train from the disjoint complement (seeds `seed`,
`seed+1000`).

- **OEQ / PAS**: 1000 test + 1000 train, uniform, with the tuning-era held-out
  ids (`data/tuning/*_test_ids.json`) excluded. The final splits may reuse
  tuning-era *training* prompts; design decisions were made on the tuning
  held-out set, which is excluded.
- **AITA**: 1000 test + 1000 train, balanced 500/500 on `is_asshole` (1 = the
  top-rated reply's verdict was YTA), excluding the tuning-era AITA held-out and
  all 1591 FLIP ids (FLIP originals are AITA posts).
- **AITA-NTA-FLIP**: 1000 test + 500 train pairs, uniform, from the full
  1591-pair released set with no exclusion. Each pair yields two prompts
  (original, flipped), fed independently.

## Caveats

- **FLIP overlaps the tuning-era held-out**: 941 of the 1500 FLIP pairs (633
  test + 308 train) were in the old tuning held-out, because the released FLIP
  set has only 1591 pairs. Treat FLIP as a fixed consistency benchmark.
- **FLIP "ERROR" rows**: the source ELEPHANT file has the literal flipped
  narration `ERROR` for 4 train pairs (`d700x0, apa38v, cnftn5, bfacqq`) and 8
  test pairs (`e6v2gx, a9mcn3, dpppvj, ctwk69, bfw5pa, e6u28z, dveorg, bc1v84`).
  They are kept as used; pairs built from the "ERROR" prompt are removed from
  the released pair files.
- **PAS near-duplicate**: train `pas_01a2c64d` equals test `pas_d92572cf` up
  to whitespace (the source ids hash exact text). Kept as used in training;
  pairs built from it are removed from the released pair files.
