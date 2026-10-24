#!/usr/bin/env python3
"""Run one frozen full-pool Drop-only budget branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    run = args.run_root.resolve()
    require(Path(__file__).resolve() == run / "source" / Path(__file__).name, "Execute frozen source")
    protocol, manifest = read_json(run / "protocol.json"), read_jsonl(run / "training_manifest.jsonl")
    require(sha256(run / "training_manifest.jsonl") == protocol["training_manifest_sha256"], "Manifest hash mismatch")
    require(0 <= args.index < len(manifest), "Index out of range")
    row = manifest[args.index]
    score_file, output = Path(row["score_file"]), Path(row["output_dir"])
    require(sha256(score_file) == row["score_file_sha256"], "Score hash mismatch")
    require(not output.exists(), "Refusing to overwrite output")
    provenance = run / "provenance" / f"branch_{args.index:02d}"
    provenance.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(run / "source/train_original_rsp_raw.py"),
        "--dataset", row["dataset"], "--data_dir", row["data_dir"],
        "--score_file", row["score_file"], "--output_dir", row["output_dir"],
        "--method", row["score_method"], "--budget", str(row["budget"]), "--seed", str(row["seed"]),
        "--model_name", "google/medgemma-4b-it", "--cache_dir", "model_cache",
        "--max_length", "1024", "--train_batch_size", "1", "--eval_batch_size", "4",
        "--gradient_accumulation_steps", "8", "--epochs", "2.0", "--learning_rate", "1e-4",
        "--lora_rank", "16", "--lora_alpha", "32", "--eval_splits", "dev", "test",
        "--selection_order", "pool_index", "--skip_adapter_save"]
    write_json(provenance / "invocation.json", {"created_utc": now(), "row": row,
        "command": command, "shell_display": shlex.join(command),
        "environment": {key: os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "CUDA_VISIBLE_DEVICES")}})
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with (provenance / "stdout.log").open("x") as stdout, (provenance / "stderr.log").open("x") as stderr:
            completed = subprocess.run(command, cwd=run / "source", stdout=stdout, stderr=stderr, check=False)
        require(completed.returncode == 0, f"Trainer exited {completed.returncode}")
        result = read_json(output / "result.json")
        require(result["budget_items"] == row["budget"], "Budget mismatch")
        require(result["selection_training_order"] == "pool_index", "Wrong training order")
        require(result["selected_uid_sha256"] == row["expected_selected_uid_sha256"], "Selected-set hash mismatch")
        write_json(provenance / "success.json", {"status": "success", "completed_utc": now(),
            "result_sha256": sha256(output / "result.json"), "predictions_sha256": sha256(output / "predictions.npz")})
    except BaseException as error:
        write_json(provenance / "failure.json", {"status": "failure", "failed_utc": now(), "error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
