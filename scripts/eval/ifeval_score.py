"""Score IFEval responses with Google's official programmatic checkers
(vendored verbatim in vendor/instruction_following_eval/).

Reports the four standard numbers -- prompt-level strict / loose (all of a
prompt's instructions pass) and instruction-level strict / loose -- with Wald
95% half-widths, plus their mean. The HF dataset pads every kwargs dict with
all possible keys set to None; those entries are dropped before the checkers
build their descriptions, matching Google's official input_data.jsonl.

Responses are joined to instructions by the exact original prompt string.
By default every prompt must have a response.

Determinism: Google's `keywords:letter_frequency` checker replaces a
non-letter target (e.g. "#" or "!") with `random.choice(ascii_letters)`, and
the language checks use `langdetect`, which is randomized unless seeded.
Unseeded, the research scorer flipped ~1 prompt between runs (e.g. 445 vs 446
of 541 prompt-level strict for base Qwen3-8B). This script seeds both
(random.seed(0) before scoring, DetectorFactory.seed = 0), so a rescore is
reproducible. The released IFEval results were produced unseeded, so a rescore
can differ from them by about one prompt.

Outputs (--out-dir): summary_<tag>.json and perprompt_<tag>.csv (key,
n_instructions, strict_all, loose_all, prompt[:200]); the per-prompt strict
column is what the paired McNemar test against the base model uses.
"""

import argparse
import csv
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plurpo.paths import REPO_ROOT, VENDOR_DIR

sys.path.insert(0, VENDOR_DIR)
from langdetect import DetectorFactory  # noqa: E402
from instruction_following_eval import evaluation_lib  # noqa: E402

DetectorFactory.seed = 0

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ifeval_generate import DEFAULT_DATA, load_records  # noqa: E402


def load_inputs(path):
    """IFEval records as evaluation_lib.InputExample, None kwargs dropped."""
    inputs = []
    for ex in load_records(path):
        kwargs = [{k: v for k, v in d.items() if v is not None} for d in ex["kwargs"]]
        assert len(kwargs) == len(ex["instruction_id_list"]), f"{ex['key']}: kwargs mismatch"
        inputs.append(evaluation_lib.InputExample(
            key=ex["key"], instruction_id_list=ex["instruction_id_list"],
            prompt=ex["prompt"], kwargs=kwargs))
    return inputs


def load_responses(paths):
    """{original prompt: response} over one or more jsonl shards."""
    out = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                if o["prompt"] in out:
                    assert out[o["prompt"]] == o["response"], f"conflicting responses in {path}"
                out[o["prompt"]] = o["response"]
    return out


def wald_ci(k, n):
    """(mean, 95% Wald half-width) for k of n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    return p, 1.96 * math.sqrt(p * (1.0 - p) / n)


def aggregate(outputs):
    """(prompts passing all, n prompts, instructions passing, n instructions)."""
    prompt_ok = sum(1 for o in outputs if o.follow_all_instructions)
    inst_ok = sum(sum(1 for x in o.follow_instruction_list if x) for o in outputs)
    inst_tot = sum(len(o.follow_instruction_list) for o in outputs)
    return prompt_ok, len(outputs), inst_ok, inst_tot


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--responses", nargs="+", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--dataset", default=DEFAULT_DATA)
    p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "outputs", "ifeval", "scored"))
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()

    random.seed(0)
    inputs = load_inputs(args.dataset)
    responses = load_responses(args.responses)
    covered = [inp for inp in inputs if inp.prompt in responses]
    missing = len(inputs) - len(covered)
    assert args.allow_partial or missing == 0, f"{missing}/{len(inputs)} prompts have no response"
    inputs = covered
    strict = [evaluation_lib.test_instruction_following_strict(i, responses) for i in inputs]
    loose = [evaluation_lib.test_instruction_following_loose(i, responses) for i in inputs]
    sp_ok, sp_n, si_ok, si_n = aggregate(strict)
    lp_ok, lp_n, li_ok, li_n = aggregate(loose)
    res = {}
    for name, (k, n) in {"prompt_strict": (sp_ok, sp_n), "prompt_loose": (lp_ok, lp_n),
                         "inst_strict": (si_ok, si_n), "inst_loose": (li_ok, li_n)}.items():
        m, ci = wald_ci(k, n)
        res[name] = {"acc": m, "ci95": ci, "k": k, "n": n}
    res["ifeval_avg"] = sum(res[m]["acc"] for m in
                            ("prompt_strict", "prompt_loose", "inst_strict", "inst_loose")) / 4.0

    os.makedirs(args.out_dir, exist_ok=True)
    sum_path = os.path.join(args.out_dir, f"summary_{args.tag}.json")
    with open(sum_path, "w") as f:
        json.dump({"tag": args.tag, "dataset": args.dataset, "n_prompts": len(inputs),
                   "n_missing": missing, **res}, f, indent=2)
    with open(os.path.join(args.out_dir, f"perprompt_{args.tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "n_instructions", "strict_all", "loose_all", "prompt"])
        for inp, s, lo in zip(inputs, strict, loose):
            w.writerow([inp.key, len(inp.instruction_id_list), int(s.follow_all_instructions),
                        int(lo.follow_all_instructions), inp.prompt[:200]])
    for key in ("prompt_strict", "prompt_loose", "inst_strict", "inst_loose"):
        r = res[key]
        print(f"  {key:14s} = {r['acc']:.3f} +/- {r['ci95']:.3f}  ({r['k']}/{r['n']})")
    print(f"  IFEval average = {res['ifeval_avg']:.3f}\n  -> {sum_path}")


if __name__ == "__main__":
    main()
