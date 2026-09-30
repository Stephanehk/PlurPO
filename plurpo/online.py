"""The iterative (on-policy) training loop shared by PlurPO stage 1, its
no-stakeholder-preferences ablation, and DPO-Neutral.

One *epoch* (the paper's term) = rebuild the preference dataset from fresh
samples of the current policy, then one RPO/DPO pass over it:

  for each chunk of prompts (breadth-first, so every stage's LLM calls fan
  out concurrently and vLLM batches them):
    A. judge.prime_panels     stakeholder panels (cached across epochs)
    B. sampler.sample         k2 candidates per prompt from the policy
    C. judge.veto_batch       vetoed / excluded candidates per prompt
    D. pairs                  every vetoed x every accepted candidate
                              (or, in the ablation, pairwise preferences)
    E. persist                one record per prompt -> responses_iter_<k>.jsonl
  then train_on_pairs; snapshot adapter_iter_<k> (+ .complete sentinel).

`judge` exposes prime_panels(posts, concurrency), panel(post) and
veto_batch(items, concurrency) -> (vetoed, excluded); see
plurpo.stakeholders.StakeholderSimulator and plurpo.neutrality.NeutralityJudge.

Restarting a killed run resumes from the last complete adapter_iter_<k>;
a partially sampled epoch is re-sampled from scratch.
"""

import gc
import json
import os

import torch

from plurpo.concurrency import concurrent_map
from plurpo.pairs import stage1_pairs
from plurpo.rpo import adapter_complete, assert_adapter_finite, save_adapter, train_on_pairs

# Share of unparsable A/B verdicts tolerated by the preference ablation; more
# means the labeler is not answering the question and the run is invalid.
PREF_UNPARSED_MAX_FRAC = 0.20


def preference_pair_indices(k, max_pairs):
    """(i, j), i < j, in offset-major order (all adjacent pairs, then offset 2,
    ...), truncated to `max_pairs` when > 0, so a capped subset covers all k
    candidates nearly evenly. Deterministic."""
    pairs = [(i, i + off) for off in range(1, k) for i in range(k - off)]
    return pairs[:max_pairs] if max_pairs else pairs


def detect_resume_point(output_dir, num_iterations):
    """Number of finished epochs (contiguous complete adapter_iter_<k>)."""
    k_max = 0
    for k in range(1, num_iterations + 1):
        if not adapter_complete(os.path.join(output_dir, f"adapter_iter_{k}")):
            break
        k_max = k
    for k in range(k_max + 2, num_iterations + 1):
        assert not adapter_complete(os.path.join(output_dir, f"adapter_iter_{k}")), (
            f"gap in finished epochs under {output_dir}")
    return k_max


class OnlineTrainer:
    """Iterative preference optimisation of `model` (a PeftModel, trainable).

    pair_source: "veto" (PlurPO / DPO-Neutral) or "preference" (ablation:
        vetoes skipped, pairs from judge.prefer over up to
        `preference_max_pairs` candidate pairs per prompt).
    sync_policy(adapter_dir): hot-loads the current adapter into the vLLM
        engine the sampler draws from.
    dpo_kwargs: plurpo.rpo.dpo_config_kwargs(...) output.
    """

    def __init__(self, model, tokenizer, judge, sampler, sync_policy, dpo_kwargs,
                 pair_source="veto", preference_max_pairs=10, chunk_size=64,
                 concurrency=128):
        assert pair_source in ("veto", "preference")
        self.model = model
        self.tokenizer = tokenizer
        self.judge = judge
        self.sampler = sampler
        self.sync_policy = sync_policy
        self.dpo_kwargs = dpo_kwargs
        self.pair_source = pair_source
        self.preference_max_pairs = preference_max_pairs
        self.chunk_size = chunk_size
        self.concurrency = concurrency
        self.pref_labeled = 0
        self.pref_unparsed = 0

    def _preference_pairs(self, post, candidates):
        """Ablation pairs over distinct candidates; presentation order of A/B
        alternates by pair index to balance position bias."""
        candidates = list(dict.fromkeys(candidates))
        pairs = []
        if len(candidates) < 2:
            return pairs
        for n, (i, j) in enumerate(preference_pair_indices(len(candidates),
                                                          self.preference_max_pairs)):
            a, b = (candidates[i], candidates[j]) if n % 2 == 0 else (candidates[j], candidates[i])
            choice = self.judge.prefer(post, a, b)
            self.pref_labeled += 1
            if choice is None:
                self.pref_unparsed += 1
                continue
            chosen, rejected = (a, b) if choice == "A" else (b, a)
            pairs.append({"prompt": post, "chosen": chosen, "rejected": rejected})
        return pairs

    def _chunk_records(self, chunk):
        """Stages A-D for one chunk -> per-prompt (record, pairs)."""
        self.judge.prime_panels(chunk, self.concurrency)
        sampled = concurrent_map(lambda x: self.sampler.sample(x, self.judge.panel(x)),
                                 chunk, self.concurrency)
        if self.pair_source == "veto":
            vetoed, excluded = self.judge.veto_batch(
                [(x, s[0]) for x, s in zip(chunk, sampled)], self.concurrency)
            pref = [[] for _ in chunk]
        else:
            vetoed = [[] for _ in chunk]
            excluded = [[] for _ in chunk]
            pref = concurrent_map(lambda a: self._preference_pairs(a[0], a[1][0]),
                                  list(zip(chunk, sampled)), self.concurrency)
        out = []
        for i, x in enumerate(chunk):
            responses, sources = sampled[i]
            pairs, non_vetoed = stage1_pairs(x, responses, vetoed[i], excluded[i])
            pairs.extend(pref[i])
            record = {"prompt": x, "responses": responses, "sources": sources,
                      "vetoed": vetoed[i], "excluded": excluded[i], "pref_pairs": pref[i]}
            out.append((record, pairs, non_vetoed))
        return out

    def build_epoch(self, prompts, epoch_idx, output_dir):
        """Sample, veto and pair every prompt for one epoch. Persists every
        prompt's record (including all-vetoed / zero-vetoed ones) before
        returning. Returns (pairs, stats)."""
        path = os.path.join(output_dir, f"responses_iter_{epoch_idx + 1}.jsonl")
        pairs, all_vetoed, zero_vetoed, contributing = [], [], [], 0
        with open(path, "w") as f:
            for start in range(0, len(prompts), self.chunk_size):
                chunk = prompts[start:start + self.chunk_size]
                for off, (record, prompt_pairs, non_vetoed) in enumerate(
                        self._chunk_records(chunk)):
                    idx = start + off
                    f.write(json.dumps(dict({"iter": epoch_idx, "prompt_idx": idx},
                                            **record)) + "\n")
                    f.flush()
                    if not record["vetoed"]:
                        zero_vetoed.append(idx)
                    if not non_vetoed:
                        all_vetoed.append(idx)
                    if prompt_pairs:
                        contributing += 1
                        pairs.extend(prompt_pairs)
                print(f"[epoch {epoch_idx + 1}] {start + len(chunk)}/{len(prompts)} prompts, "
                      f"{len(pairs)} pairs", flush=True)
        if self.pref_labeled:
            frac = self.pref_unparsed / self.pref_labeled
            assert frac <= PREF_UNPARSED_MAX_FRAC, (
                f"{frac:.1%} of preference verdicts unparsable "
                f"({self.pref_unparsed}/{self.pref_labeled})")
        stats = {"iter": epoch_idx, "num_pairs": len(pairs),
                 "contributing_prompts": contributing,
                 "all_vetoed_prompt_indices": all_vetoed,
                 "zero_vetoed_prompt_indices": zero_vetoed}
        return pairs, stats

    def train(self, prompts, output_dir, num_iterations, start_iteration=0):
        """Run epochs start_iteration..num_iterations-1; save adapter_iter_<k>
        after each and the final adapter at `output_dir`."""
        stats_path = os.path.join(output_dir, "iter_stats.json")
        iter_stats = []
        if start_iteration > 0:
            with open(stats_path) as f:
                iter_stats = json.load(f)[:start_iteration]
        for k in range(start_iteration, num_iterations):
            gc.collect()
            torch.cuda.empty_cache()
            self.model.eval()
            sync_dir = os.path.join(output_dir, "policy_current")
            assert_adapter_finite(self.model, f"policy sync -> {sync_dir}")
            self.model.save_pretrained(sync_dir)
            self.sync_policy(sync_dir)
            pairs, stats = self.build_epoch(prompts, k, output_dir)
            print(f"[epoch {k + 1}/{num_iterations}] pairs={stats['num_pairs']} "
                  f"contributing={stats['contributing_prompts']}/{len(prompts)} "
                  f"all_vetoed={len(stats['all_vetoed_prompt_indices'])} "
                  f"zero_vetoed={len(stats['zero_vetoed_prompt_indices'])}", flush=True)
            iter_stats.append(stats)
            if pairs:
                train_on_pairs(self.model, self.tokenizer, pairs,
                               os.path.join(output_dir, f"iter_{k + 1}"), self.dpo_kwargs)
            else:
                print(f"[epoch {k + 1}] no pairs; skipping the update")
            save_adapter(self.model, os.path.join(output_dir, f"adapter_iter_{k + 1}"),
                         f"epoch {k + 1} snapshot")
            with open(stats_path, "w") as f:
                json.dump(iter_stats, f, indent=2)
        save_adapter(self.model, output_dir, "final adapter")
        with open(stats_path, "w") as f:
            json.dump(iter_stats, f, indent=2)
