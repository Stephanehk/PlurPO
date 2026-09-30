"""PlurPO stage 2, step 1: sample candidates from the stage-1 policy.

For every prompt of the TRAIN split (never the held-out split; or of an
explicit --prompts-file) draw --n samples
from base + the stage-1 LoRA on a local vLLM engine. AITA-NTA-FLIP contributes
both narrations of each pair, keyed "<id>::orig" / "<id>::flip".

Generation length is 1536 tokens: these are training targets and must be
complete; `finished` records whether the sample hit EOS (step 3 drops
truncated samples).

Output: <out-dir>/pairs/candidates.jsonl, one flushed row per sample
  {"id", "sample_idx", "sentence", "response", "finished"}
A rerun skips prompts whose samples are all on disk.
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plurpo.data import ENVS, keyed_prompts, load_split
from plurpo.io import read_jsonl
from plurpo.models import DEFAULT_MODEL_ID
from plurpo.vllm_server import launch_vllm_server, load_lora_adapter, wait_for_vllm_server

LORA_NAME = "policy"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True, choices=ENVS)
    ap.add_argument("--adapter", default=None,
                    help="stage-1 LoRA adapter dir; omit to sample the bare base model")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=0,
                    help="vLLM context window, 0 = model default (use 8192 for Qwen3-32B)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N prompts only")
    ap.add_argument("--prompts-file", default=None,
                    help="JSON [[id, text], ...] to sample instead of the env's train "
                         "split (used by the reliability experiment)")
    args = ap.parse_args()
    if args.adapter is not None:
        assert os.path.exists(os.path.join(args.adapter, "adapter_config.json")), args.adapter

    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [(str(i), t) for i, t in json.load(f)]
        assert len({i for i, _ in prompts}) == len(prompts), "duplicate ids in --prompts-file"
    else:
        prompts = keyed_prompts(args.env, load_split(args.env, "train"))
    if args.limit:
        prompts = prompts[:args.limit]
    out_path = os.path.join(args.out_dir, "pairs", "candidates.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        done = {(str(r["id"]), int(r["sample_idx"])) for r in read_jsonl(out_path)}
    todo = [(i, p) for i, p in prompts if any((i, k) not in done for k in range(args.n))]
    print(f"[sample] {args.env}: {len(prompts)} prompts, {len(todo)} to sample")
    if not todo:
        return

    log = os.path.join(args.out_dir, "vllm_sample.log")
    proc, url = launch_vllm_server(args.model_id, args.device,
                                   max_model_len=args.max_model_len or None,
                                   log_path=log, enable_lora=args.adapter is not None,
                                   max_lora_rank=16)
    wait_for_vllm_server(proc, url, log_path=log)
    serve_name = args.model_id
    if args.adapter is not None:
        load_lora_adapter(url, LORA_NAME, args.adapter)
        serve_name = LORA_NAME
    from openai import OpenAI
    client = OpenAI(base_url=url + "/v1", api_key="EMPTY", timeout=1200)
    lock = threading.Lock()
    fh = open(out_path, "a")

    def run_prompt(item):
        pid, text = item
        resp = client.chat.completions.create(
            model=serve_name, messages=[{"role": "user", "content": text}],
            max_tokens=args.max_tokens, n=args.n, temperature=args.temperature,
            top_p=args.top_p, seed=args.seed,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        rows = [{"id": pid, "sample_idx": k, "sentence": text,
                 "response": ch.message.content.strip(),
                 "finished": ch.finish_reason == "stop"}
                for k, ch in enumerate(resp.choices) if ch.message.content is not None]
        with lock:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.flush()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(run_prompt, todo))
    fh.close()
    proc.terminate()
    print(f"[sample] done -> {out_path}")


if __name__ == "__main__":
    main()
