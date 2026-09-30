"""Generate responses to the 541 IFEval prompts (Zhou et al., 2023) for one
model configuration: the base model or base + a trained LoRA adapter.

Used to check that PlurPO does not degrade general instruction following
(appendix IFEval tables). Scoring is programmatic (scripts/eval/ifeval_score.py);
no LM judge is involved.

Decoding: GREEDY (the standard IFEval setting), max 1280 new tokens, same chat
template path as every other generation (plurpo.models.build_chat_input). The
response file records the ORIGINAL prompt, which is IFEval's join key.

Data: data/ifeval/ifeval_data.jsonl (a verbatim dump of google/IFEval, so no
hub access is needed).

Output: <out-dir>/responses_ifeval_<tag>_greedy_seed0[_shard{i}of{N}].jsonl,
one {"key", "prompt", "gen_prompt", "response"} object per line.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from plurpo.generation import attach_adapter, generate_one, shard_slice
from plurpo.models import DEFAULT_MODEL_ID, load_model
from plurpo.paths import DATA_DIR, REPO_ROOT

DEFAULT_DATA = os.path.join(DATA_DIR, "ifeval", "ifeval_data.jsonl")


def load_records(path):
    """IFEval records (key, prompt, instruction_id_list, kwargs), file order."""
    with open(path) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    assert recs, f"no records in {path}"
    return recs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--base", action="store_true")
    src.add_argument("--adapter", default=None)
    p.add_argument("--tag", required=True, help="output stem, e.g. base or plurpo_oeq")
    p.add_argument("--dataset", default=DEFAULT_DATA)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default=None)
    p.add_argument("--max-new-tokens", type=int, default=1280)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "outputs", "ifeval", "responses"))
    args = p.parse_args()

    items = [(r["key"], r["prompt"]) for r in load_records(args.dataset)]
    shard_tag = ""
    if args.num_shards > 1:
        s, e = shard_slice(len(items), args.num_shards, args.shard_id)
        items = items[s:e]
        shard_tag = f"_shard{args.shard_id}of{args.num_shards}"
    print(f"[ifeval] prompts={len(items)} tag={args.tag}")
    model, tokenizer = load_model(args.model_id, device=args.device,
                                  cache_dir=args.cache_dir, device_map=args.device_map)
    if args.adapter:
        model = attach_adapter(model, args.adapter)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir, f"responses_ifeval_{args.tag}_greedy_seed{args.seed}{shard_tag}.jsonl")
    with open(out_path, "w") as f:
        for k, (key, prompt) in enumerate(items):
            resp = generate_one(model, tokenizer, prompt, args.max_new_tokens,
                                1.0, 1.0, greedy=True)
            f.write(json.dumps({"key": key, "prompt": prompt, "gen_prompt": prompt,
                                "response": resp}) + "\n")
            f.flush()
            if (k + 1) % 50 == 0:
                print(f"[ifeval] {k + 1}/{len(items)}", flush=True)
    print(f"[ifeval] wrote {out_path}")


if __name__ == "__main__":
    main()
