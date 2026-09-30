"""PlurPO stage 2, step 2: the frozen model labels each candidate's voice.

A local vLLM server with the BASE model (M2; the same model family as the
policy, e.g. Qwen3-8B for Qwen3-8B and Qwen3-32B for Qwen3-32B) answers
OUTSIDER / STAKEHOLDER / UNCLEAR for every candidate, greedily, using the
rubric in plurpo/prompts/role_confusion.py.

Label mapping: OUTSIDER -> 0 (chosen-eligible), STAKEHOLDER -> 2
(rejected-eligible), anything else -> -1 (dropped).

Output: <out-dir>/pairs/labels.csv with columns id, sample_idx, label,
addressee (always -1: not judged), voice. Resumable.
"""

import argparse
import csv
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plurpo.data import ENVS
from plurpo.io import read_keyed_candidates
from plurpo.models import DEFAULT_MODEL_ID
from plurpo.prompts.role_confusion import build_voice_messages, parse_voice, voice_to_label
from plurpo.vllm_server import launch_vllm_server, wait_for_vllm_server


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, help="dir holding pairs/candidates.jsonl")
    ap.add_argument("--env", required=True, choices=ENVS)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="the frozen judge")
    ap.add_argument("--max-model-len", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    cands = read_keyed_candidates(os.path.join(args.out_dir, "pairs", "candidates.jsonl"))
    out_path = os.path.join(args.out_dir, "pairs", "labels.csv")
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {(r["id"], int(r["sample_idx"])) for r in csv.DictReader(f)}
    todo = [(k, v) for k, v in sorted(cands.items()) if k not in done]
    print(f"[label] {len(cands)} candidates, {len(todo)} to label")
    if not todo:
        return

    log = os.path.join(args.out_dir, "vllm_voiceonly.log")
    proc, url = launch_vllm_server(args.model_id, args.device,
                                   max_model_len=args.max_model_len or None,
                                   log_path=log, enable_lora=False)
    wait_for_vllm_server(proc, url, log_path=log)
    from openai import OpenAI
    client = OpenAI(base_url=url + "/v1", api_key="EMPTY", timeout=600)

    new = not os.path.exists(out_path)
    fh = open(out_path, "a", newline="")
    writer = csv.writer(fh)
    if new:
        writer.writerow(["id", "sample_idx", "label", "addressee", "voice"])
    lock = threading.Lock()

    def run_one(kv):
        (pid, idx), row = kv
        system, user = build_voice_messages(args.env, row["sentence"], row["response"])
        out = client.chat.completions.create(
            model=args.model_id,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=8, temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        voice = parse_voice(out.choices[0].message.content or "")
        with lock:
            writer.writerow([pid, idx, voice_to_label(voice), -1, voice])
            fh.flush()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(run_one, todo))
    fh.close()
    proc.terminate()
    print(f"[label] done -> {out_path}")


if __name__ == "__main__":
    main()
