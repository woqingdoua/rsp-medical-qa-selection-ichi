#!/usr/bin/env python3
"""Aggregate full-pool Drop-only budget experiment from per-question predictions."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def norm(value) -> str:
    return " ".join(str(value).strip().lower().split())


def metrics(probabilities: np.ndarray, gold: np.ndarray) -> tuple[float, float]:
    prediction = probabilities.argmax(1)
    f1 = []
    for label in range(probabilities.shape[1]):
        tp = np.sum((prediction == label) & (gold == label))
        fp = np.sum((prediction == label) & (gold != label))
        fn = np.sum((prediction != label) & (gold == label))
        denominator = 2 * tp + fp + fn
        f1.append(float(2 * tp / denominator) if denominator else 0.0)
    return float(np.mean(prediction == gold)), float(np.mean(f1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    args = parser.parse_args(); run = args.run_root.resolve()
    protocol, manifest = read_json(run / "protocol.json"), read_jsonl(run / "training_manifest.jsonl")
    output = run / "aggregation"; output.mkdir(exist_ok=False)
    values, individual, missing = {}, [], []
    for index, row in enumerate(manifest):
        branch = Path(row["output_dir"])
        if not (run / "provenance" / f"branch_{index:02d}" / "success.json").is_file():
            missing.append({"index": index, "dataset": row["dataset"], "seed": row["seed"], "budget": row["budget"]})
            continue
        result = read_json(branch / "result.json")
        if result["selected_uid_sha256"] != row["expected_selected_uid_sha256"]:
            raise ValueError(f"Selection verification failed for branch {index}")
        with np.load(branch / "predictions.npz", allow_pickle=False) as archive:
            probabilities, gold = archive["test_probs"], archive["test_gold"]
        if row["dataset"] == "medexpqa":
            pool = read_jsonl(Path(row["data_dir"]) / "pool.jsonl")
            test = read_jsonl(Path(row["data_dir"]) / "test.jsonl")
            pool_questions = {norm(item["question"]) for item in pool}
            keep = np.asarray([i for i, item in enumerate(test) if norm(item["question"]) not in pool_questions])
            probabilities, gold = probabilities[keep], gold[keep]
        accuracy, f1 = metrics(probabilities, gold)
        item = {"dataset": row["dataset"], "budget": row["budget"], "seed": row["seed"],
                "method": protocol["method"], "test_n": len(gold), "test_accuracy": accuracy, "test_macro_f1": f1,
                "selected_uid_sha256": row["expected_selected_uid_sha256"], "predictions_file": str(branch / "predictions.npz")}
        individual.append(item); values[(row["dataset"], row["budget"], row["seed"])] = (accuracy, f1)
    expected = {(d, b, s) for d in protocol["datasets"] for b in protocol["budgets"] for s in protocol["seeds"]}
    if set(values) != expected:
        (output / "completion.json").write_text(json.dumps({"status": "partial", "missing": missing}, indent=2) + "\n")
        raise SystemExit("Incomplete full-pool Drop-only experiment")
    summary = []
    for dataset in (*protocol["datasets"], "dataset_macro"):
        for budget in protocol["budgets"]:
            per_seed = []
            for seed in protocol["seeds"]:
                if dataset == "dataset_macro":
                    per_seed.append(tuple(float(np.mean([values[(d, budget, seed)][m] for d in protocol["datasets"]])) for m in (0, 1)))
                else:
                    per_seed.append(values[(dataset, budget, seed)])
            array = np.asarray(per_seed)
            summary.append({"dataset": dataset, "budget": budget, "method": protocol["method"], "n_seeds": len(per_seed),
                "test_accuracy_mean": float(array[:, 0].mean()), "test_accuracy_sd": float(array[:, 0].std(ddof=1)),
                "test_macro_f1_mean": float(array[:, 1].mean()), "test_macro_f1_sd": float(array[:, 1].std(ddof=1))})
    for name, records in (("individual_results.csv", individual), ("summary_results.csv", summary)):
        with (output / name).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0].keys()); writer.writeheader(); writer.writerows(records)
    (output / "completion.json").write_text(json.dumps({"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "expected_branches": len(expected), "completed_branches": len(values), "missing": []}, indent=2) + "\n")
    print((output / "completion.json").read_text())


if __name__ == "__main__":
    main()
