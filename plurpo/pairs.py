"""Preference-pair construction rules for both PlurPO training stages.

Stage 1 (Sec. 3.1, step 4): a response accepted by ALL simulated stakeholders is
preferred to every response vetoed by AT LEAST ONE. Responses whose veto status
could not be read (an unparsable stance or vote, recorded as `excluded`) are
neither chosen nor rejected. Pairs are emitted per occurrence and not deduped:
k vetoed and m accepted responses yield k * m pairs.

Stage 2 (role-confusion mitigation, App. "Mitigating PlurPO role confusion"):
within one prompt, a candidate the frozen model labels as spoken in a
stakeholder's voice (label 2) is rejected against a uniformly drawn candidate
labelled as an assistant/outsider voice (label 0). Both sides must be complete
(generation hit EOS), at most `max_pairs_per_prompt` rejected per prompt.

Both functions are pure: no model calls, no global RNG.
"""

import random
import statistics


def stage1_pairs(prompt, responses, vetoed, excluded):
    """Stage-1 pairs for one prompt.

    Inputs: the bare `prompt`; all sampled `responses` (in sampling order); the
    `vetoed` subset (in response order); the `excluded` subset. Membership is by
    string equality, so if two identical samples differ in veto status both
    copies leave the chosen pool.
    Returns (pairs, non_vetoed): `pairs` is a list of
    {"prompt", "chosen", "rejected"} dicts, ordered rejected-major.
    """
    vetoed_set = set(vetoed)
    excluded_set = set(excluded)
    non_vetoed = [r for r in responses
                  if r not in vetoed_set and r not in excluded_set]
    pairs = []
    for rejected in vetoed:
        for chosen in non_vetoed:
            pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return pairs, non_vetoed


def pairs_from_record(record):
    """Rebuild one prompt's stage-1 pairs from a persisted
    `responses_iter_<k>.jsonl` record (keys prompt/responses/vetoed/excluded).
    Valid because the stakeholder simulator is frozen: the record fully
    determines the pairs. Returns the pair list."""
    pairs, _non_vetoed = stage1_pairs(record["prompt"], record["responses"],
                                      record["vetoed"], record.get("excluded", []))
    return pairs


def role_confusion_pairs(candidates, labels, max_pairs_per_prompt, seed,
                         require_finished=True):
    """Stage-2 (role-confusion) pairs.

    Inputs:
      candidates: dict (prompt_id, sample_idx) -> row with keys
        sentence, response, finished; iteration order = file order with
        last-write-wins (as produced by `load_jsonl_keyed`).
      labels: dict (prompt_id, sample_idx) -> int label; 0 = assistant voice
        (chosen-eligible), 1 or 2 = collapsed voice (rejected-eligible),
        -1 = unparsed (dropped, never guessed).
      max_pairs_per_prompt, seed, require_finished: see module docstring.
    Returns (pairs, manifest). Deterministic given the inputs and `seed`: prompts
    are visited in sorted-id order and one seeded RNG drives every draw.
    """
    by_prompt = {}
    for key, row in candidates.items():
        lab = labels.get(key)
        if lab is None:
            continue
        g = by_prompt.setdefault(key[0], {"sentence": row["sentence"], "chosen": [],
                                          "rejected": [], "n_seen": 0, "n_trunc": 0})
        g["n_seen"] += 1
        if require_finished and not row.get("finished", False):
            g["n_trunc"] += 1
            continue
        if lab == 0:
            g["chosen"].append(row["response"])
        elif lab in (1, 2):
            g["rejected"].append(row["response"])

    rng = random.Random(seed)
    pairs = []
    n_no_rejected = n_no_chosen = n_capped = 0
    clen, rlen = [], []
    for pid in sorted(by_prompt):
        g = by_prompt[pid]
        if not g["rejected"]:
            n_no_rejected += 1
            continue
        if not g["chosen"]:
            n_no_chosen += 1
            continue
        rejected = g["rejected"]
        if len(rejected) > max_pairs_per_prompt:
            rejected = rng.sample(rejected, max_pairs_per_prompt)
            n_capped += 1
        for rej in rejected:
            ch = rng.choice(g["chosen"])
            pairs.append({"prompt": g["sentence"], "chosen": ch, "rejected": rej})
            clen.append(len(ch))
            rlen.append(len(rej))
    rng.shuffle(pairs)

    lab_counts = {}
    for v in labels.values():
        lab_counts[v] = lab_counts.get(v, 0) + 1
    manifest = {
        "n_candidates": len(candidates),
        "n_labelled": len(labels),
        "label_counts": {str(k): v for k, v in sorted(lab_counts.items())},
        "n_prompts_seen": len(by_prompt),
        "n_prompts_contributing": len(by_prompt) - n_no_rejected - n_no_chosen,
        "n_prompts_dropped_no_failure": n_no_rejected,
        "n_prompts_dropped_no_clean": n_no_chosen,
        "n_prompts_hit_pair_cap": n_capped,
        "n_truncated_dropped": sum(g["n_trunc"] for g in by_prompt.values()),
        "max_pairs_per_prompt": max_pairs_per_prompt,
        "require_finished": bool(require_finished),
        "seed": seed,
        "n_pairs": len(pairs),
        "chosen_length": _length_stats(clen),
        "rejected_length": _length_stats(rlen),
    }
    return pairs, manifest


def _length_stats(xs):
    """Character-length summary for the manifest (DPO can exploit a systematic
    chosen/rejected length gap, so the balance is always reported)."""
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean_chars": round(statistics.mean(xs), 1),
            "median_chars": statistics.median(xs)}
