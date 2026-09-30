# data/

Everything needed to train and evaluate PlurPO without the raw source files
(~485 MB).

| dir | contents | size |
|---|---|---|
| `splits/` | frozen train / evaluation splits for OEQ, PAS, AITA, AITA-NTA-FLIP (see its README) | 16 MB |
| `tuning/` | held-out OEQ / PAS split used only to tune PlurPO | 3.9 MB |
| `ifeval/` | `ifeval_data.jsonl`, a verbatim dump of google/IFEval (541 prompts) | 0.6 MB |
| `plurpo_qwen3-8b_pairs/` | stage-1 PlurPO pairs built by Qwen3-8B, used to train Qwen3-32B, gzipped (see its README) | 45 MB (447 MB unpacked) |
| `qwen3-8b_role_confusion_pairs/` | stage-2 (role-confusion) pairs of the Qwen3-8B PlurPO models | 15 MB |

Integrity checks: `pytest tests/test_data_splits.py tests/test_data_pairs.py`.

## Sources and licensing

All prompts come from the ELEPHANT social-sycophancy benchmark (Cheng et al.,
2025, arXiv:2505.13995) and the OSF bundle of Cheng et al. (2025,
arXiv:2510.01395; https://osf.io/smvw7). Use is subject to those releases'
terms; please cite them.

| env | raw source used by the research code |
|---|---|
| OEQ | OSF `Data/OEQ_results/endorse4_elephant_oeq3k_human.csv` (prompt + top human response; 3010 prompts after dedup) |
| PAS | OSF `Data/PAS_results/endorse4_elephant_oeq_pas_claude.csv` (6395 prompts) |
| AITA | ELEPHANT r/AmITheAsshole posts with the top reply's verdict (79,990-post pool) |
| AITA-NTA-FLIP | ELEPHANT `AITA-NTA-FLIP.csv` (1591 original / perspective-flipped pairs) |

Row ids are the research code's ids (OEQ/PAS: a hash of the prompt text; AITA /
FLIP: the reddit post id).

## tuning/ — the PlurPO tuning split

`oeq_{train,test}` and `pas_{train,test}` (1000 + 1000 each; same columns as
`splits/`; load with `load_split(env, split, split_set="tuning")`). This is the
seed-0 split PlurPO's prompts and hyper-parameters were tuned on; the paper
never re-uses its held-out half. It produced:

- the Iterative-RPO alpha sweep (App. "Tuning the Iterative RPO alpha
  parameter"): trained on `tuning/oeq_train`, scored on all 1000 prompts of
  `tuning/oeq_test`; the human baseline in that table is `tuning/oeq_test`'s
  `human_response`.
- the GEPA baselines' validation set (`tuning/oeq_test`); GEPA's training set is
  `splits/oeq_train`.

`tuning/*_test` is id-disjoint from both `splits/*` halves; up to
whitespace/case, 3 OEQ tuning-test prompts reappear under other ids
(`oeq_402994ee`~train `oeq_3fbe2e76`; `oeq_a10f81de`~test `oeq_a50ef2e1`;
`oeq_a0cbb986`~test `oeq_c1bdbea0`). `tuning/*_train` is not disjoint: 503 OEQ (184 PAS) of its prompts are in `splits/*_train` and 494 OEQ (179
PAS) are in `splits/*_test`, because the final split only excluded the tuning
held-out.

Several research artifacts are tagged "origdata" although they used the
**final** split: the Inferred-Prefs-DPO arms (trained on `splits/*_train`, all
four alpha variants scored on `splits/*_test`) and the in-context-examples
arms (exemplar pool `splits/oeq_train`; k = 2, 5, 10 all scored on
`splits/oeq_test`). So every Inferred-Prefs-DPO and in-context number is on the
same held-out set as the paper's other results. Note that the k = 5 choice was
therefore made on `splits/oeq_test`, not on a separate held-out set as the paper
states.

## Role-confusion (stage-2) pairs

`qwen3-8b_role_confusion_pairs/<env>_pairs.jsonl`: `{"prompt", "chosen",
"rejected"}`. For each training prompt, 10 candidates were sampled from the
Qwen3-8B stage-1 PlurPO adapter, the frozen base model labelled each as assistant/outsider voice
(chosen-eligible) or stakeholder voice (rejected), and
`plurpo.pairs.role_confusion_pairs(max_pairs_per_prompt=10, seed=0)` built the
pairs. Each file regenerates byte-for-byte from the run's candidates and labels.
AITA / AITA-NTA-FLIP pairs use the verdict-aware voice rubric (the "R2"
relabel). `<env>_pairs_manifest.json` carries the builder's counts plus a
`release_clean` block.

The Qwen3-32B role-confusion pairs are not included in this release.

| env | pairs |
|---|---|
| OEQ | 448 |
| PAS | 2016 |
| AITA | 328 |
| AITA-NTA-FLIP | 701 |
