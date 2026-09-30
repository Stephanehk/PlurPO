"""Regression test: PlurPO stage-1 epoch construction is unchanged from the code
that produced the paper's results.

`fixtures/stage1_equivalence_digests.json` holds sha256 digests of what the
ORIGINAL research implementation produced for the first 14 training prompts of
an env under the deterministic fake LLM in fake_llm.py: every LLM prompt sent
to the frozen model (with its token cap), every policy sampling call, every
persisted record (candidates, sources, vetoed, excluded), the pairs, and the
epoch statistics. This test rebuilds the same epoch with the released code and
requires byte-identical output, for each paper configuration (Qwen3 on three
envs, Phi-4, Llama-3.1-8B, Granite-4.1-8B, and the
no-stakeholder-preferences ablation).

Runs on CPU in seconds; no model, GPU, or network needed.
"""

import hashlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from fake_llm import FakeLLM, FakePolicy
from plurpo.candidates import CandidateSampler
from plurpo.data import load_split, train_prompts
from plurpo.online import OnlineTrainer
from plurpo.stakeholders import StakeholderSimulator

CFGS = {
    "qwen3-8b_OEQ": dict(env="OEQ", veto_prompt="P6", user=True, n_model=2, n_steer=2, assess=True, pref=False),
    "qwen3-8b_PAS": dict(env="PAS", veto_prompt="P6", user=True, n_model=2, n_steer=2, assess=False, pref=False),
    "qwen3-8b_FLIP": dict(env="AITA-NTA-FLIP", veto_prompt="P6", user=True, n_model=2, n_steer=2, assess=True, pref=False),
    "phi-4_AITA": dict(env="AITA", veto_prompt="P6", user=True, n_model=4, n_steer=0, assess=False, pref=False),
    "llama_OEQ": dict(env="OEQ", veto_prompt="P6_stance_only", user=False, n_model=4, n_steer=0, assess=False, pref=False),
    "granite_PAS": dict(env="PAS", veto_prompt="P6_granite", user=False, n_model=4, n_steer=0, assess=False, pref=False),
    "pairpref_OEQ": dict(env="OEQ", veto_prompt="P6", user=True, n_model=2, n_steer=2, assess=True, pref=True),
}
KEEP = ("prompt_idx", "prompt", "responses", "sources", "vetoed", "excluded", "pref_pairs")


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _build(cfg):
    fake, pol = FakeLLM(), FakePolicy()
    judge = StakeholderSimulator(fake, veto_prompt=cfg["veto_prompt"], user_stakeholder=cfg["user"],
                                 stakeholder_max_tokens=1024, veto_max_tokens=128)
    sampler = CandidateSampler(pol, n_samples=10, n_model=cfg["n_model"], n_steer=cfg["n_steer"],
                               steer="perspshift", perspshift_assess=cfg["assess"])
    trainer = OnlineTrainer(None, None, judge, sampler, None, None,
                            pair_source="preference" if cfg["pref"] else "veto",
                            preference_max_pairs=10, chunk_size=3, concurrency=4)
    prompts = train_prompts(cfg["env"], load_split(cfg["env"], "train"))[:14]
    out_dir = tempfile.mkdtemp()
    pairs, stats = trainer.build_epoch(prompts, 0, out_dir)
    with open(os.path.join(out_dir, "responses_iter_1.jsonl")) as f:
        records = [{k: json.loads(line)[k] for k in KEEP} for line in f]
    llm_calls = {}
    for c in fake.calls:
        llm_calls[c] = llm_calls.get(c, 0) + 1
    pol_calls = {}
    for c in pol.calls:
        pol_calls[c] = pol_calls.get(c, 0) + 1
    return {
        "pairs": pairs,
        "records": records,
        "stats": {"all_vetoed": stats["all_vetoed_prompt_indices"],
                  "zero_vetoed": stats["zero_vetoed_prompt_indices"],
                  "contributing": stats["contributing_prompts"]},
        "llm_calls": sorted([[p, m, n] for (p, m), n in llm_calls.items()]),
        "policy_calls": sorted([list(k) + [n] for k, n in pol_calls.items()], key=str),
    }


def test_stage1_matches_original_implementation():
    with open(os.path.join(HERE, "fixtures", "stage1_equivalence_digests.json")) as f:
        expected = json.load(f)
    assert set(expected) == set(CFGS)
    for name, cfg in CFGS.items():
        got = _build(cfg)
        for key, digest in expected[name].items():
            assert _digest(got[key]) == digest, f"{name}: {key} differs from the original"


if __name__ == "__main__":
    test_stage1_matches_original_implementation()
    print("ok")
