"""Release convenience launcher for the ICHI paper's frozen recipe.

Unlike the archival utilities, this interface fixes paper-specific defaults.
It does not run anything on import. GPU commands require an explicit subcommand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def invoke(script, values):
    subprocess.run([sys.executable, str(ROOT / "src" / script), *map(str, values)], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("score", "train"):
        c = sub.add_parser(name)
        c.add_argument("--dataset", required=True, choices=("afrimedqa", "medexpqa", "medexqa", "pubmedqa", "medmcqa"))
        c.add_argument("--data-dir", type=Path, required=True)
        c.add_argument("--output-dir", type=Path, required=True)
        c.add_argument("--cache-dir", type=Path, default=Path("model_cache"))
        c.add_argument("--budget", choices=(64, 128, 256), type=int, default=128)
        c.add_argument("--seed", choices=(13, 23, 37), type=int, required=True)
    sub.choices["score"].add_argument("--full-pool", action="store_true")
    tr = sub.choices["train"]
    tr.add_argument("--selection-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists():
        p.error("Refusing to overwrite an existing run")
    common = ["--dataset", a.dataset, "--data_dir", a.data_dir, "--cache_dir", a.cache_dir,
              "--budget", a.budget, "--seed", a.seed, "--max_length", 1024]
    if a.command == "score":
        fixed = ["--noise_layer", -1, "--noise_sigma", 0.02, "--mc_passes", 8,
                 "--batch_size", 4, "--prefilter_mult", 4,
                 "--prefilter_name", "none" if a.full_pool else "answer_entropy"]
        invoke("score_original_rsp_raw.py", [*common, *fixed, "--phase", "clean", "--output_dir", a.output_dir / "clean"])
        # This script's legacy `rsp` column is a placeholder. Always derive the
        # automatic coefficient with selection.py; do not use that column.
        invoke("score_original_rsp_raw.py", [*common, *fixed, "--phase", "rsp", "--drop_weight", 0,
                                            "--clean_dir", a.output_dir / "clean", "--output_dir", a.output_dir / "components"])
    else:
        manifest = json.loads((a.selection_dir / "selection.json").read_text())
        if manifest["budget"] != a.budget:
            p.error("Selection budget does not match training budget")
        with (a.data_dir / "pool.jsonl").open() as handle:
            pool = [json.loads(line) for line in handle if line.strip()]
        indices = manifest["selected_indices"]
        expected = [pool[i]["uid"] for i in indices]
        if expected != manifest["selected_uids"]:
            p.error("Selected UID/order does not match the supplied bank")
        with (a.selection_dir / "scores.jsonl").open() as handle:
            score_rows = [json.loads(line) for line in handle if line.strip()]
        if [r["uid"] for r in score_rows] != [r["uid"] for r in pool]:
            p.error("Complete score/bank UID order mismatch")
        ranked = sorted(range(len(pool)), key=lambda i: (-float(score_rows[i][manifest["method"]]), i))
        if sorted(ranked[:a.budget]) != indices:
            p.error("Score file does not reproduce frozen membership")
        invoke("train_original_rsp_raw.py", [*common, "--score_file", a.selection_dir / "scores.jsonl",
            "--output_dir", a.output_dir, "--method", manifest["method"], "--selection_order", "pool_index",
            "--train_batch_size", 1, "--eval_batch_size", 4, "--gradient_accumulation_steps", 8,
            "--epochs", 2, "--learning_rate", 1e-4, "--lora_rank", 16, "--lora_alpha", 32,
            "--eval_splits", "dev", "test", "--skip_adapter_save"])


if __name__ == "__main__":
    main()
