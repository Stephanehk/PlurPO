# PlurPO: Pluralistic Preference Optimization

Code, data and scored outputs for *Mitigating Language Model Sycophancy in
Interpersonal Settings*.

PlurPO trains an LM to give responses acceptable to the stakeholders of a
user's situation. A frozen copy of the model identifies those stakeholders and
simulates each one's veto or acceptance of candidate responses; responses
accepted by all are preferred to responses vetoed by any, and the model is
updated with Iterative RPO on these pairs (Sec. 3). A second stage removes role
confusion (the model answering in the voice of a stakeholder).

```
PlurPO/
├── plurpo/            library (data, models, simulator, sampling, RPO, judges, metrics)
│   └── prompts/       every prompt, copied verbatim from the code that produced the paper
├── configs/           per-model recipes: qwen3-8b, qwen3-32b, phi-4, llama-3.1-8b, granite-4.1-8b
├── scripts/
│   ├── train/         PlurPO stage 1 & 2, Qwen3-32B transfer, pair rebuilding
│   ├── baselines/     DPO-Neutral, Inferred-Prefs-DPO, GEPA
│   ├── eval/          generation (+ prompting baselines, rejection sampling), LM judges, IFEval, reliability
│   └── analysis/      every table and figure, computed from results/
├── slurm/             job templates and end-to-end drivers
├── data/              splits, tuning split, IFEval prompts, the PlurPO datasets (see data/README.md)
├── results/           scored held-out outputs behind every number in the paper (see results/README.md)
├── vendor/            third-party code used verbatim (ELEPHANT judges, IFEval)
└── tests/
```

## Setup

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt          # Qwen3 models, judges, analysis, baselines
# Phi-4 / Llama-3.1-8B / Granite-4.1-8B were trained and sampled in a newer
# stack: requirements-other-families.txt (Python 3.12, CUDA 13)
export HF_HOME=/path/to/hf_cache         # base models: Qwen/Qwen3-8B, Qwen/Qwen3-32B, ...
export OPENROUTER_API_KEY=...            # only for the gpt-5-mini judges and DPO-Neutral / GEPA
python -m pytest tests/                  # CPU-only; no model, GPU or API needed
```

All scripts run from any directory. Training needs NVIDIA GPUs (2 x 80 GB for
stage 1); the LM judges call `openai/gpt-5-mini` through OpenRouter.

## Reproducing the paper's numbers (no GPU, no API)

```bash
python scripts/analysis/reproduce_all.py        # tables + figures -> results/_reproduced/
python -m pytest tests/test_reproduce_main.py    # asserts the main-text values
```

| paper item | script |
|---|---|
| Fig. 2 (OEQ/PAS endorsement) + samples panels | `scripts/analysis/figures/endorsement.py` |
| Fig. 3 (AITA / AITA-Flipped verdict errors) | `scripts/analysis/figures/verdict_errors.py` |
| Table 1 (explicit-stance fraction) | `scripts/analysis/tables_main.py` |
| Fig. 4 + App. ELEPHANT figures | `scripts/analysis/figures/elephant.py` |
| App. original-judge, macro-F1, samples | `scripts/analysis/figures/{v1_judge,aita_f1_samples}.py` |
| App. neutral-rate table | `scripts/analysis/tables_main.py` |
| App. veto-vs-endorse table | `scripts/analysis/tables_appendix.py` |

Due to file-size constraints, this repository does not include
the Qwen3-32B results (`results/qwen3-32b/`) or the results for the other model
families (Phi-4, Llama-3.1-8B, Granite-4.1-8B). They are available on request.
The first-draw, reliability and IFEval outputs behind the remaining appendix
tables and the stronger-teacher figures are likewise not included (see
`results/README.md`).

## Training PlurPO

Recipes are in `configs/`. One environment (`OEQ`, `PAS`, `AITA`,
`AITA-NTA-FLIP`) per model, trained on `data/splits/<env>_train.csv`.

```bash
# Stage 1: 3 epochs of simulate-stakeholders -> veto -> pairs -> RPO (2 GPUs)
python scripts/train/train_plurpo.py --config qwen3-8b --env OEQ \
    --output-dir runs/plurpo_qwen3-8b_oeq_stage1
# Stage 2: role-confusion mitigation (1 GPU) -> final adapter
python scripts/train/run_role_confusion.py --config qwen3-8b --env OEQ \
    --adapter runs/plurpo_qwen3-8b_oeq_stage1 --out-dir runs/plurpo_qwen3-8b_oeq
# or both as a Slurm chain:
bash slurm/run_plurpo.sh qwen3-8b OEQ runs
```

The same commands train Phi-4, Llama-3.1-8B and Granite-4.1-8B with
`--config phi-4|llama-3.1-8b|granite-4.1-8b` (`slurm/run_other_families.sh`).

**Qwen3-32B from the Qwen3-8B PlurPO dataset** (`data/plurpo_qwen3-8b_pairs/`):
`bash slurm/run_qwen3-32b.sh OEQ runs` (one offline RPO epoch over all three
8B epochs' pairs from a fresh LoRA, then stage 2 with Qwen3-32B as its own
judge).

Stage-1 outputs include `responses_iter_<k>.jsonl` with every candidate, its
source, and its veto/exclusion status; `scripts/train/rebuild_pairs.py`
rebuilds the exact training pairs from them.

## Evaluation

`slurm/run_eval_all.sh` runs the whole protocol for one model on one env: the
base model, the six prompting baselines (`--steer`, `--perspective-shift`) and a
trained adapter are sampled on the 1,000 held-out prompts; the rejection
sampling judge re-samples responses that do not engage with the user (up to 5
attempts); then the action-endorsement judge (OEQ/PAS), the verdict classifier
(AITA/FLIP) and the ELEPHANT judges score the kept responses.
`scripts/analysis/build_frames.py` joins these outputs into the `results/`
layout the analysis scripts read. IFEval: `slurm/eval_ifeval.sbatch`.

## Baselines and ablations

| paper | driver |
|---|---|
| prompting baselines, Perspective-shift, consider-stakeholders prompt | `slurm/run_eval_all.sh`, `slurm/run_stakeholder_prompt.sh` |
| DPO-Neutral | `slurm/run_dpo_neutral.sh` |
| Inferred-Prefs-DPO | `slurm/run_inferred_prefs.sh` |
| GEPA (per-response, endorsement-rate) | `slurm/run_gepa.sh` |
| in-context examples | `slurm/run_kshot.sh` |
| PlurPO-no-stakeholder-preferences | `slurm/run_pairpref.sh` |
| Iterative-RPO alpha sweep (tuning split) | `slurm/run_alpha_sweep.sh` |
| per-epoch evaluation | `slurm/run_per_epoch_eval.sh` |
| consistency (reliability) | `slurm/run_reliability.sh` |
