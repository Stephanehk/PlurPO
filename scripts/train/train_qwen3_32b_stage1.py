"""Qwen3-32B stage 1 from the Qwen3-8B PlurPO dataset (Sec. 4.4, App.
"Training Qwen3-32B with the preference dataset constructed by PlurPO and
Qwen3-8B").

Creates a fresh LoRA on Qwen3-32B, then runs ONE offline RPO epoch over all
three Qwen3-8B epochs' stage-1 pairs for the env
(data/plurpo_qwen3-8b_pairs/<stem>_pairs.jsonl.gz). Stage 2 then runs with
run_role_confusion.py --config qwen3-32b, i.e. candidates sampled from and
labelled by Qwen3-32B itself.

Usage:
  python scripts/train/train_qwen3_32b_stage1.py --env OEQ \
      --out-dir runs/plurpo_qwen3-32b_oeq_stage1
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from plurpo.config import load_config, per_env
from plurpo.data import ENVS, STEM
from plurpo.paths import DATA_DIR


def run(script, *args):
    cmd = [sys.executable, os.path.join(HERE, script)] + [str(a) for a in args]
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True, choices=ENVS)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    args = ap.parse_args()
    cache = ["--cache-dir", args.cache_dir] if args.cache_dir else []
    cfg = load_config("qwen3-32b")
    o = cfg["offline_stage1"]
    pairs = os.path.join(DATA_DIR, "plurpo_qwen3-8b_pairs", f"{STEM[args.env]}_pairs.jsonl.gz")
    assert os.path.exists(pairs), pairs
    init = args.out_dir.rstrip("/") + "_init"
    if not os.path.exists(os.path.join(init, "adapter_config.json")):
        run("make_fresh_lora.py", "--model-id", cfg["model_id"], "--out-dir", init,
            *cache)
    device_map = per_env(o["device_map"], args.env)
    extra = ["--device-map", device_map] if device_map else []
    run("train_offline.py", "--adapter", init, "--out-dir", args.out_dir,
        "--pairs-file", pairs, "--model-id", cfg["model_id"],
        *cache, "--beta", o["beta"],
        "--rpo-alpha", o["rpo_alpha"], "--learning-rate", o["learning_rate"],
        "--num-train-epochs", o["num_train_epochs"], "--grad-accum", o["grad_accum"],
        "--max-length", o["max_length"],
        "--precompute-ref-batch-size", o["precompute_ref_batch_size"],
        "--seed", o["dpo_seed"], *extra)


if __name__ == "__main__":
    main()
