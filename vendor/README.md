# Vendored third-party code

Copied verbatim (no edits) so the evaluation is reproducible offline.

| Directory | Source | License | Used by |
|---|---|---|---|
| `elephant/sycophancy_scorers.py` | https://github.com/myracheng/elephant (commit `61f7044fc0e11defc971dfaa9bc84d7340fa88eb`), Cheng et al. 2025, *ELEPHANT: Measuring and understanding social sycophancy in LLMs* | CC0 1.0 (`elephant/LICENSE`) | `scripts/eval/judge_elephant.py` — the validation and framing judge prompts are built by its `create_prompt` |
| `instruction_following_eval/` | google-research/google-research `instruction_following_eval` (Zhou et al. 2023, IFEval) | Apache-2.0 (headers in each file) | `scripts/eval/ifeval_score.py` — the official strict/loose checkers |

`sycophancy_scorers.py` builds an OpenAI client at import time and reads
`OPENAI_API_KEY`; `judge_elephant.py` sets a placeholder before importing it and
never uses that client (all judging goes through OpenRouter).
