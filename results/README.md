# results/: scored outputs behind every table and figure

The Qwen3-8B main-text numbers and figures (plus the ELEPHANT-judge,
original-judge, macro-F1 / samples and veto-vs-endorse appendix items) can be
regenerated from this directory alone, with no GPU and no API calls:

    python scripts/analysis/reproduce_all.py      # -> results/_reproduced/{tables,figures}
    python -m pytest tests/test_reproduce_main.py # main-text numbers == paper

Figure PNGs are written under the paper's `\includegraphics` file names and
are **pixel-identical** to the published figures, with two exceptions that
correct defects in the originals (see "Deviations from the paper" below).

Due to file-size constraints, this repository does not include
the Qwen3-32B results (`results/qwen3-32b/`) or the results for the other model
families (Phi-4, Llama-3.1-8B, Granite-4.1-8B). They are available on request.

Also not included, for the same reason: the first-draw (no rejection sampling) frames behind the per-epoch, alpha-sweep,
role-confusion, Inferred-Prefs-DPO, in-context and ablation tables and the
stronger-teacher figures; the reliability samples; and the IFEval generations
and scores.

Result frames do not carry the prompt text (`sentence`, `post_text`,
`flipped_post_text`); join on `id` with `data/splits/<env>_test.csv`.

## Rebuilding a frame from a fresh evaluation run

`_provenance/` only runs against our original research artifacts. To turn a
NEW run of the evaluation pipeline (`scripts/eval/generate.py`, `gate.py`,
`generate.py --gate-loop`, `judge_endorsement.py`, `classify_verdicts.py`,
`judge_elephant.py`, `judge_role_confusion.py`) into the same frames, use
`scripts/analysis/build_frames.py`, one arm at a time:

    # OEQ / PAS, rejection-sampled frame -> results/<model>/<ENV>/<arm>.csv
    python scripts/analysis/build_frames.py --model qwen3-8b --env OEQ --arm plurpo \
        --run-dir <run> --tag final --scores-root <scores> \
        --endorse-name <attempt-1 --name> --endorse-gateloop-name <gateloop --name> \
        [--v1-name <gated v1 --name>] \
        [--elephant-dir <scores>/elephant/OEQ --elephant-name <name> --elephant-gateloop-name <name>]
    # AITA / AITA-NTA-FLIP (verdicts come from eval/aita_scored_* in the run dir)
    python scripts/analysis/build_frames.py --model qwen3-8b --env AITA-NTA-FLIP --arm base \
        --run-dir <run> --tag base [--elephant-dir ... --elephant-name ... --elephant-gateloop-name ...]
    # OEQ human reference
    python scripts/analysis/build_frames.py --model qwen3-8b --env OEQ --arm human \
        --scores-root <scores> --endorse-name <human --name>

`--tag` is the adapter tag (`final`, `base`, `base-becritical`, ...); the
`_finaldata_temp1.0_seed0` suffix is added. Scored files are joined by `id`
(and `side`) when present, otherwise positionally, with prompt/response text
asserted to line up. `tests/test_build_frames.py` checks that this script
rebuilds the shipped qwen3-8b OEQ (base, PlurPO, human) and AITA-NTA-FLIP
(base, PlurPO) frames cell-for-cell from the
original run directories (skipped where those are not mounted).

## Layout

| path | contents |
|---|---|
| `<model>/<ENV>/<arm>.csv` | held-out responses **after rejection sampling** (the paper's protocol), one row per prompt, with every judge label. Index: `arms.csv` |
| `veto_vs_endorse/` | 500 sampled training candidates per PlurPO epoch with the stakeholder veto and the endorsement label |
| `_provenance/` | the scripts that built the frames above from the original research artifacts (kept for auditability; they read `PLURPO_RESEARCH_ROOT`, a directory holding the original `runs/` and `sycoharm_runs/`; recorded paths use the prefix `<research>`) |
| `MANIFEST.csv` | sha256 + original location (under `<research>`) of every file |

`model` is `qwen3-8b`.
`ENV` is `OEQ`, `PAS`, `AITA` or `AITA-NTA-FLIP` (AITA-Flipped).

### Arm keys

| arm | paper name |
|---|---|
| `base` | the untrained model (e.g. "Qwen3-8B") |
| `becritical`, `dontbesyco`, `explicit`, `sharma` | + "Be critical", + "Don't-be-sycophantic", + Cheng et al. (2026) prompt, + Sharma et al. (2024) prompt |
| `perspshift` | Perspective-shift |
| `dpo_neutral` | DPO-Neutral (the neutral-mixed DPO run: half the candidates sampled with the neutral instruction) |
| `dpo_neutral_nonmixed` | Neutral DPO without the steered candidates (see deviation 3) |
| `plurpo` | PlurPO (stage 1 + role-confusion stage) |
| `plurpo_stage1` | PlurPO after stage 1 only |
| `human` | OEQ human responses (never gated) |

## Frame schema

Rejection-sampled frames (`<model>/<ENV>/<arm>.csv`):

* **OEQ / PAS**: `id, response, endorse_response, gate_passed, n_tried`,
  plus `framing_response, validation_response` (ELEPHANT judges) and
  `endorse_v1_response` (original Cheng et al. judge prompt).
* **AITA**: `id, response, is_asshole, pred, gate_passed, n_tried`
  (+ `framing_response, validation_response`).
* **AITA-NTA-FLIP**: `id, response_original, response_flipped, pred_original,
  pred_flipped, gate_passed_original, gate_passed_flipped, n_tried_original,
  n_tried_flipped`
  (+ `framing_original/_flipped, validation_original/_flipped`).

Conventions:
* `endorse_response`: gpt-5-mini action-endorsement label, 0 = challenges,
  1 = explicitly endorses, 2 = implicitly endorses, 3 = neutral. Action
  endorsement rate = #1 / (#0 + #1).
* `pred`: gpt-5-mini verdict classifier, `YTA` / `NTA` / `UNCLEAR`.
* `gate_passed = 0`: no response passed the engagement filter within 5 draws;
  the prompt is dropped from every rate and its response/labels are empty.
  `n_tried` = number of draws taken (1 = passed first time). The OEQ human
  reference was never gated (`n_tried` empty).
* AITA drops UNCLEAR verdicts from the FNR/FPR denominators; AITA-Flipped keeps
  UNCLEAR pairs in the both-NTA/both-YTA denominator; a FLIP pair is dropped if
  either side failed rejection sampling.

## Which script produces what

| paper item | script (under `scripts/analysis/`) | frames |
|---|---|---|
| Fig. 2, App. full OEQ/PAS panels | `figures/endorsement.py`, `tables_main.py` (`endorsement_qwen3-8b_*`) | rejection-sampled |
| Table 1 (explicit stance) | `tables_main.py` (`explicit_stance`) | rejection-sampled |
| Fig. 3 (AITA / AITA-Flipped) | `figures/verdict_errors.py`, `tables_main.py` (`verdict_qwen3-8b_*`) | rejection-sampled |
| Fig. 4 + App. ELEPHANT judges | `figures/elephant.py`, `tables_main.py` (`elephant_*`) | rejection-sampled |
| App. neutral-rate table | `tables_main.py` (`neutral_rate_*`) | rejection-sampled |
| App. original-judge figures | `figures/v1_judge.py` | rejection-sampled |
| App. AITA macro-F1, AITA/FLIP samples | `figures/aita_f1_samples.py` | rejection-sampled |
| App. veto-vs-endorse | `tables_appendix.py` | `veto_vs_endorse/` |

## Deviations from the paper (found while reproducing)

Figures that change when regenerated from data:
1. **`flip_samples.png`**: the published mean-samples values for the in-loop
   AITA-Flipped baselines double-counted resampled sides (the gate loop
   resampled BOTH sides of a pair when either failed, and those extra draws were
   counted). Corrected means: Qwen3-8B 1.004 -> 1.000, Cheng 1.002 -> 1.001,
   Sharma 1.005 -> 1.001, Don't-be-sycophantic 1.007 -> 1.003, Be critical
   1.005 -> 1.001, Perspective-shift 1.016 -> 1.006 (PlurPO unchanged, 1.004).
2. **`aita_samples.png`**: the published version averaged over YTA/NTA-classified
   rows only; the regenerated version averages over every scored prompt. The
   printed values are identical to 3 decimals; one error bar shifts by a few pixels.

Numbers in the paper text that do not match the data (the data are shipped as-is):
3. **Two different "DPO-Neutral" arms.** The endorsement figures and tables use
   the neutral-mixed run (`dpo_neutral`), but the ELEPHANT-judge figures
   (Fig. 4 and App.) use the non-mixed run (`dpo_neutral_nonmixed`, e.g. PAS
   76.5% / 99.6%). Reproduced as published (`common.ELEPHANT_ARMS`). With the
   neutral-mixed run the PAS row would read 78.7% framing / 93.2% validation.
4. **Table 1, PlurPO OEQ = 0.282** comes from the first-draw PlurPO responses
   (282/1000; not included in this release). Every other cell of Table 1 and
   Fig. 2 is rejection-sampled, under which it is 0.284 [0.256, 0.312].

Missing outputs (never produced in the research runs; not fabricated):
* Crisis ("harm") judge figures are commented out in the paper and not
  regenerated.
