"""Held-out generation for OEQ / PAS / AITA / AITA-NTA-FLIP, with in-loop
rejection sampling.

TWO MODES
---------
1. Attempt-1 generation (default). One response per held-out prompt of the
   frozen final TEST split, from the base model (`--base`, optionally with a
   prompting-baseline steer) or a LoRA adapter (`--adapter`, default: the run
   dir's own final adapter). Shardable; `torch.manual_seed(--seed)` once per
   shard.
2. Rejection sampling (`--gate-loop --worklist W`). For every prompt the gate
   judge rejected at attempt 1 (see scripts/eval/gate.py, which writes W),
   resample from the SAME arm (identical steer / rewrite / adapter) and judge
   each draw inline with the gpt-5-mini gate, stopping at the first pass or
   after MAX_ATTEMPTS=5 total attempts. Draw k is seeded by
   attempt_seed(post, k, seed). Needs OPENROUTER_API_KEY on the GPU node.
   For AITA-NTA-FLIP both sides of a worklist pair are resampled; downstream
   code treats the attempt-1 gate labels as the authority on which side was
   actually rejected.

OUTPUTS (under <run-dir>/eval/, tag = <adapter_tag>_finaldata_temp1.0_seed0)
  OEQ/PAS   responses_<tag>[_shard{i}of{N}].csv   id, sentence, response,
            winning_attempt, gate_passed, n_tried  (shard suffix only if N>1)
            responses_<tag>[_shard..].jsonl       + gen_prompt, rewritten
  AITA      aita_responses_<tag>_shard{i}of{N}.csv (suffix always)
            id, post_text, rewritten_post, response, winning_attempt,
            gate_passed, n_tried, is_asshole
  FLIP      aita_responses_<tag>_shard{i}of{N}.csv  id, post_text,
            flipped_post_text, rewritten_{original,flipped},
            response_{original,flipped}, winning_attempt_*, gate_passed_*,
            n_tried_*
  gate loop eval/gate/responses_<tag>_gateloop.csv (OEQ/PAS) or
            eval/gate/aita_responses_<tag>_gateloop.csv (AITA/FLIP), same
            columns, only the worklist prompts.
  + summary_[aita_]<tag>*.json with the full configuration.

Decoding defaults match the paper: temperature 1.0, top_p 0.95, 512 new tokens
for OEQ/PAS and 1536 for AITA/FLIP; perspective-shift rewrites are greedy with
1536 tokens. The scored `sentence` / `post_text` is always the bare original
post.
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import torch

from plurpo.data import AITA_ENVS, ENVS, load_split
from plurpo.generation import (MAX_ATTEMPTS, adapter_tag_for, attach_adapter,
                               build_tag, generate_one, resample_until_pass,
                               shard_slice, steer_suffix_tag)
from plurpo.models import DEFAULT_MODEL_ID, load_model
from plurpo.perspective_shift import perspective_shift_post
from plurpo.prompts.baselines import STEERS

DEFAULT_SHARDS = {"OEQ": 2, "PAS": 2, "AITA": 1, "AITA-NTA-FLIP": 2}
DEFAULT_MAX_TOKENS = {"OEQ": 512, "PAS": 512, "AITA": 1536, "AITA-NTA-FLIP": 1536}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, choices=ENVS)
    p.add_argument("--run-dir", required=True,
                   help="outputs go to <run-dir>/eval/; also the default --adapter")
    p.add_argument("--base", action="store_true", help="untrained base model, no adapter")
    p.add_argument("--adapter", default=None, help="LoRA adapter dir (default: --run-dir)")
    p.add_argument("--steer", default="none", choices=sorted(STEERS),
                   help="prompting-baseline instruction (plurpo/prompts/baselines.py)")
    p.add_argument("--perspective-shift", action="store_true",
                   help="answer the greedy third-person rewrite of each post")
    p.add_argument("--assess-narrative", action="store_true",
                   help="with --perspective-shift, append 'Assess this narrative.' "
                        "(the paper's Perspective-shift baseline uses it on all envs)")
    p.add_argument("--k-shot-human", type=int, default=None,
                   help="OEQ only: k (post, human response) exemplar turns from the "
                        "final train split before the held-out post")
    p.add_argument("--exemplar-seed", type=int, default=0)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default=None, help="e.g. 'auto' for Qwen3-32B")
    p.add_argument("--num-shards", type=int, default=None,
                   help="default: 2 for OEQ/PAS/FLIP, 1 for AITA (the paper layout)")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--n", type=int, default=None, help="debug: first N prompts only")
    p.add_argument("--sample-max-tokens", type=int, default=None,
                   help="default: 512 for OEQ/PAS, 1536 for AITA/FLIP")
    p.add_argument("--rewrite-max-tokens", type=int, default=1536)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-set", default="final", choices=("final", "tuning"),
                   help="'tuning' = the held-out OEQ/PAS split (alpha sweep)")
    p.add_argument("--gate-loop", action="store_true",
                   help="rejection-sample the prompts in --worklist (see module doc)")
    p.add_argument("--worklist", default=None, help="gate worklist json from gate.py")
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    args = p.parse_args()
    assert not (args.base and args.adapter), "--base and --adapter are mutually exclusive"
    n_steers = sum([args.steer != "none", args.perspective_shift, bool(args.k_shot_human)])
    assert n_steers <= 1, "--steer / --perspective-shift / --k-shot-human are exclusive"
    assert not (args.assess_narrative and not args.perspective_shift), (
        "--assess-narrative only applies with --perspective-shift")
    assert not (args.k_shot_human and args.env != "OEQ"), (
        "--k-shot-human needs paired human responses, which only OEQ has")
    assert args.gate_loop == bool(args.worklist), "--gate-loop and --worklist go together"
    if args.num_shards is None:
        args.num_shards = 1 if args.gate_loop else DEFAULT_SHARDS[args.env]
    assert not (args.gate_loop and args.num_shards != 1), "the gate loop runs unsharded"
    if args.sample_max_tokens is None:
        args.sample_max_tokens = DEFAULT_MAX_TOKENS[args.env]
    return args


def arm_tag(args):
    """adapter tag + steer suffix (see plurpo.generation for the convention)."""
    tag = adapter_tag_for(args.base, args.adapter or args.run_dir, args.run_dir)
    if args.perspective_shift:
        return tag + "-perspshift" + ("assess" if args.assess_narrative else "")
    if args.k_shot_human:
        return f"{tag}-{args.k_shot_human}shothuman"
    return tag + steer_suffix_tag(args.steer)


def select_rows(args):
    """Held-out rows in frozen id order, filtered to the worklist (gate loop)
    or sliced to this shard. Filtering happens before sharding so row order
    (and hence exemplar draws) is independent of worklist ordering."""
    rows = load_split(args.env, "test", args.split_set)
    if args.n is not None:
        rows = rows.iloc[:args.n].reset_index(drop=True)
    if args.gate_loop:
        with open(args.worklist) as f:
            want = set(json.load(f)["ids"])
        rows = rows[rows["id"].isin(want)].reset_index(drop=True)
        assert len(rows), f"worklist {args.worklist} matched 0 held-out prompts"
        print(f"[gate] resampling {len(rows)} worklist prompts")
    s, e = shard_slice(len(rows), args.num_shards, args.shard_id)
    print(f"[gen] shard {args.shard_id}/{args.num_shards}: rows [{s}:{e}) of {len(rows)}")
    return rows.iloc[s:e].reset_index(drop=True)


def output_names(args, tag):
    """(out_dir, csv, jsonl, summary) paths for this job."""
    is_aita = args.env in AITA_ENVS
    if args.gate_loop:
        out_dir = os.path.join(args.run_dir, "eval", "gate")
        stem = f"{tag}_gateloop"
    else:
        out_dir = os.path.join(args.run_dir, "eval")
        always_suffix = is_aita or args.num_shards > 1
        stem = f"{tag}_shard{args.shard_id}of{args.num_shards}" if always_suffix else tag
    os.makedirs(out_dir, exist_ok=True)
    csv_name = f"aita_responses_{stem}.csv" if is_aita else f"responses_{stem}.csv"
    jsonl_name = f"responses_aita_{stem}.jsonl" if is_aita else f"responses_{stem}.jsonl"
    summ_name = f"summary_aita_{stem}.json" if is_aita else f"summary_{stem}.json"
    return (out_dir, os.path.join(out_dir, csv_name), os.path.join(out_dir, jsonl_name),
            os.path.join(out_dir, summ_name))


def make_responder(args, model, tokenizer, gate_judge):
    """-> respond(post, exemplars) -> (response, rewritten, winning_attempt, n_tried)."""
    prefix, suffix = STEERS[args.steer]

    def draw(gen_prompt):
        return generate_one(model, tokenizer, gen_prompt, args.sample_max_tokens,
                            args.temperature, args.top_p, greedy=False)

    def respond(post, exemplars):
        rewritten = None
        if args.perspective_shift:
            rewritten = perspective_shift_post(
                lambda p: generate_one(model, tokenizer, p, args.rewrite_max_tokens,
                                       args.temperature, args.top_p, greedy=True),
                post, assess_narrative=args.assess_narrative)
        body = rewritten if rewritten is not None else post
        if exemplars is not None:
            gen_prompt = []
            for ex_post, ex_human in exemplars:
                gen_prompt.append({"role": "user", "content": ex_post})
                gen_prompt.append({"role": "assistant", "content": ex_human})
            gen_prompt.append({"role": "user", "content": body})
        else:
            gen_prompt = prefix + body + suffix
        if not args.gate_loop:
            return draw(gen_prompt), rewritten, 1, 1, gen_prompt
        resp, win, n_tried = resample_until_pass(
            lambda: draw(gen_prompt), gate_judge, post, post, args.seed,
            max_attempts=args.max_attempts, log=lambda m: print(m, flush=True))
        return resp, rewritten, win, n_tried, gen_prompt

    return respond


def main():
    args = parse_args()
    adapter_tag = arm_tag(args)
    tag = build_tag(adapter_tag, args.temperature, args.seed)
    rows = select_rows(args)
    out_dir, csv_path, jsonl_path, summary_path = output_names(args, tag)
    print(f"[gen] env={args.env} tag={tag} rows={len(rows)} -> {csv_path}")

    model, tokenizer = load_model(args.model_id, device=args.device,
                                  cache_dir=args.cache_dir, device_map=args.device_map)
    adapter_dir = None
    if not args.base:
        adapter_dir = args.adapter or args.run_dir
        model = attach_adapter(model, adapter_dir)

    gate_judge = None
    if args.gate_loop:
        from plurpo.judges import gate_one
        from plurpo.llm_api import make_client
        client = make_client()
        gate_judge = lambda context, response: gate_one(client, args.env, context, response)[0]

    exemplar_pool, exemplar_rng = None, None
    if args.k_shot_human:
        train = load_split("OEQ", "train")
        exemplar_pool = list(zip(train["post_text"], train["human_response"]))
        exemplar_rng = random.Random(args.exemplar_seed)

    respond = make_responder(args, model, tokenizer, gate_judge)
    torch.manual_seed(args.seed)
    records = []
    with open(jsonl_path, "w") as jf:
        for k, row in rows.iterrows():
            exemplars = (exemplar_rng.sample(exemplar_pool, args.k_shot_human)
                         if exemplar_pool is not None else None)
            if args.env == "AITA-NTA-FLIP":
                ro, rwo, wo, no, _ = respond(row["post_text"], exemplars)
                rf, rwf, wf, nf, _ = respond(row["flipped_post_text"], exemplars)
                rec = {"id": row["id"], "post_text": row["post_text"],
                       "flipped_post_text": row["flipped_post_text"],
                       "rewritten_original": rwo, "rewritten_flipped": rwf,
                       "response_original": ro, "response_flipped": rf,
                       "winning_attempt_original": wo, "winning_attempt_flipped": wf,
                       "gate_passed_original": wo is not None,
                       "gate_passed_flipped": wf is not None,
                       "n_tried_original": no, "n_tried_flipped": nf}
                jrec = rec
            else:
                resp, rw, win, n_tried, gen_prompt = respond(row["post_text"], exemplars)
                if args.env == "AITA":
                    rec = {"id": row["id"], "post_text": row["post_text"],
                           "rewritten_post": rw, "response": resp,
                           "winning_attempt": win, "gate_passed": win is not None,
                           "n_tried": n_tried, "is_asshole": int(row["is_asshole"])}
                    jrec = rec
                else:
                    rec = {"id": row["id"], "sentence": row["post_text"], "response": resp,
                           "winning_attempt": win, "gate_passed": win is not None,
                           "n_tried": n_tried}
                    jrec = {"id": row["id"], "prompt": row["post_text"],
                            "gen_prompt": gen_prompt, "rewritten": rw, "response": resp}
            records.append(rec)
            jf.write(json.dumps(jrec) + "\n")
            jf.flush()
            if (k + 1) % 50 == 0:
                print(f"[gen] {k + 1}/{len(rows)}", flush=True)

    pd.DataFrame(records).to_csv(csv_path, index=False)
    summary = {"env": args.env, "tag": tag, "adapter_tag": adapter_tag, "run_dir": os.path.abspath(args.run_dir),
               "adapter": None if adapter_dir is None else os.path.abspath(adapter_dir),
               "model_id": args.model_id, "steer": args.steer,
               "perspective_shift": args.perspective_shift,
               "assess_narrative": args.assess_narrative,
               "k_shot_human": args.k_shot_human, "exemplar_seed": args.exemplar_seed,
               "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
               "sample_max_tokens": args.sample_max_tokens,
               "num_shards": args.num_shards, "shard_id": args.shard_id,
               "gate_loop": args.gate_loop, "max_attempts": args.max_attempts,
               "n_rows": len(records)}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[gen] wrote {csv_path}\n[gen] wrote {jsonl_path}\n[gen] wrote {summary_path}")


if __name__ == "__main__":
    main()
