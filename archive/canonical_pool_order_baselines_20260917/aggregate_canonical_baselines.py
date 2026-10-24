#!/usr/bin/env python3
"""Aggregate canonical pool-index baseline reruns."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def norm(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def metric(probabilities, gold):
    pred = probabilities.argmax(1)
    f1 = []
    for label in range(probabilities.shape[1]):
        tp = np.sum((pred == label) & (gold == label)); fp = np.sum((pred == label) & (gold != label)); fn = np.sum((pred != label) & (gold == label))
        f1.append(float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0)
    return float(np.mean(pred == gold)), float(np.mean(f1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    args = parser.parse_args(); run = args.run_root.resolve()
    protocol, manifest = read_json(run / "protocol.json"), read_jsonl(run / "training_manifest.jsonl")
    output = run / "aggregation"; output.mkdir(exist_ok=False)
    rows, errors, values = [], [], {}
    for index, entry in enumerate(manifest):
        try:
            branch = Path(entry["output_dir"])
            result = read_json(branch / "result.json")
            if not (run / f"provenance/branch_{index:02d}/success.json").is_file(): raise ValueError("missing success receipt")
            if result["selection_training_order"] != "pool_index" or result["selected_uid_sha256"] != entry["expected_selected_uid_sha256"]: raise ValueError("selection/order mismatch")
            with np.load(branch / "predictions.npz", allow_pickle=False) as archive: probs, gold = archive["test_probs"], archive["test_gold"]
            keep = np.arange(len(gold))
            if entry["dataset"] == "medexpqa":
                pool, test = read_jsonl(Path(entry["data_dir"]) / "pool.jsonl"), read_jsonl(Path(entry["data_dir"]) / "test.jsonl")
                questions = {norm(item["question"]) for item in pool}; keep = np.asarray([i for i, item in enumerate(test) if norm(item["question"]) not in questions])
            accuracy, macro_f1 = metric(probs[keep], gold[keep]); key = (entry["dataset"], entry["seed"], entry["method_label"])
            values[key] = (accuracy, macro_f1); rows.append({"dataset": entry["dataset"], "seed": entry["seed"], "method": entry["method_label"], "test_accuracy": accuracy, "test_macro_f1": macro_f1})
        except Exception as error: errors.append({"index": index, "error": str(error)})
    expected = {(dataset, seed, method) for dataset in protocol["datasets"] for seed in protocol["seeds"] for method in protocol["methods"]}
    if set(values) != expected: errors.append({"coverage": f"expected={len(expected)}, found={len(values)}"})
    if errors:
        (output / "completion_audit.json").write_text(json.dumps({"status": "failed", "errors": errors}, indent=2) + "\n"); raise SystemExit("incomplete baseline aggregation")
    summary = []
    for dataset in (*protocol["datasets"], "dataset_macro"):
        for method in protocol["methods"]:
            per_seed = []
            for seed in protocol["seeds"]:
                per_seed.append(tuple(float(np.mean([values[(item, seed, method)][metric_index] for item in protocol["datasets"]])) for metric_index in (0, 1)) if dataset == "dataset_macro" else values[(dataset, seed, method)])
            summary.append({"dataset": dataset, "method": method, "n_seeds": len(per_seed), "test_accuracy_mean": float(np.mean([v[0] for v in per_seed])), "test_accuracy_std": float(np.std([v[0] for v in per_seed], ddof=1)), "test_macro_f1_mean": float(np.mean([v[1] for v in per_seed])), "test_macro_f1_std": float(np.std([v[1] for v in per_seed], ddof=1))})
    for name, records in (("individual_results.csv", rows), ("summary_results.csv", summary)):
        with (output / name).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0].keys()); writer.writeheader(); writer.writerows(records)
    (output / "completion_audit.json").write_text(json.dumps({"status": "complete", "validated_branches": len(values), "errors": [], "training_order": "ascending original pool index"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
