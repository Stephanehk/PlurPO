"""PlurPO stage 2 (role-confusion mitigation), end to end on one GPU:
sample -> label -> pairs -> one offline RPO epoch continuing the stage-1
adapter. Settings come from the config's "role_confusion" block; each step is
skipped if its output already exists, so a killed job can simply be rerun.

Usage:
  python scripts/train/run_role_confusion.py --config qwen3-8b --env OEQ \
      --adapter runs/plurpo_qwen3-8b_oeq_stage1 --out-dir runs/plurpo_qwen3-8b_oeq
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from plurpo.config import load_config
from plurpo.data import ENVS
from plurpo.rpo import adapter_complete


def run(script, *args):
    """Run a sibling script with the current interpreter; fail on error."""
    cmd = [sys.executable, os.path.join(HERE, script)] + [str(a) for a in args]
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--env", required=True, choices=ENVS)
    ap.add_argument("--adapter", required=True, help="finished stage-1 adapter")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    args = ap.parse_args()
    cache = ["--cache-dir", args.cache_dir] if args.cache_dir else []
    cfg = load_config(args.config)
    rc = cfg["role_confusion"]
    model = cfg["model_id"]
    assert adapter_complete(args.adapter), f"stage-1 adapter not finished: {args.adapter}"
    pdir = os.path.join(args.out_dir, "pairs")

    run("role_confusion_sample.py", "--env", args.env, "--adapter", args.adapter,
        "--out-dir", args.out_dir, "--model-id", model, "--n", rc["n"],
        "--temperature", rc["temperature"], "--top-p", rc["top_p"],
        "--max-tokens", rc["max_tokens"], "--seed", rc["seed"],
        "--max-model-len", rc["max_model_len"])
    run("role_confusion_label.py", "--out-dir", args.out_dir, "--env", args.env,
        "--model-id", model, "--max-model-len", rc["max_model_len"])
    if not os.path.exists(os.path.join(pdir, "pairs.jsonl")):
        run("role_confusion_pairs.py", "--out-dir", args.out_dir,
            "--max-pairs-per-prompt", rc["max_pairs_per_prompt"], "--seed", rc["seed"])
    if adapter_complete(args.out_dir):
        print(f"[role-confusion] {args.out_dir} already trained")
        return
    extra = ["--device-map", rc["device_map"]] if rc["device_map"] else []
    run("train_offline.py", "--adapter", args.adapter, "--out-dir", args.out_dir,
        "--model-id", model, *cache,
        "--beta", rc["beta"], "--rpo-alpha", rc["rpo_alpha"],
        "--learning-rate", rc["learning_rate"], "--grad-accum", rc["grad_accum"],
        "--max-length", rc["max_length"],
        "--precompute-ref-batch-size", rc["precompute_ref_batch_size"],
        "--seed", rc["dpo_seed"], *extra)


if __name__ == "__main__":
    main()
