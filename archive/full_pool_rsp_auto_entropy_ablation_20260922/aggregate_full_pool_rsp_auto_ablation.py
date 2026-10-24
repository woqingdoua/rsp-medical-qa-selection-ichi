#!/usr/bin/env python3
"""Aggregate matched full-pool automatic-lambda0 RSP entropy ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


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


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


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


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--prefilter_results", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_root.resolve()
    protocol = read_json(run / "protocol.json")
    manifest = read_jsonl(run / "training_manifest.jsonl")
    if sha256(run / "training_manifest.jsonl") != protocol["training_manifest_sha256"]:
        raise ValueError("Training manifest hash mismatch")
    output = run / "aggregation"
    output.mkdir(exist_ok=False)

    full_rows = []
    missing = []
    for index, row in enumerate(manifest):
        branch = Path(row["output_dir"])
        success = run / "provenance" / f"branch_{index:02d}" / "success.json"
        if not success.is_file() or not (branch / "predictions.npz").is_file():
            missing.append({"index": index, "dataset": row["dataset"], "budget": row["budget"], "seed": row["seed"]})
            continue
        result = read_json(branch / "result.json")
        if result["selected_uid_sha256"] != row["expected_selected_uid_sha256"]:
            raise ValueError(f"Selection hash mismatch for branch {index}")
        with np.load(branch / "predictions.npz", allow_pickle=False) as archive:
            probabilities = archive["test_probs"]
            gold = archive["test_gold"]
        data_dir = Path(row["data_dir"])
        test = read_jsonl(data_dir / "test.jsonl")
        if row["dataset"] == "medexpqa":
            pool = read_jsonl(data_dir / "pool.jsonl")
            pool_questions = {norm(item["question"]) for item in pool}
            keep = np.asarray([index for index, item in enumerate(test) if norm(item["question"]) not in pool_questions])
            if len(keep) != 121:
                raise ValueError("Unexpected MedExpQA filtered test size")
            probabilities, gold = probabilities[keep], gold[keep]
        accuracy, macro_f1 = metrics(probabilities, gold)
        full_rows.append({
            "dataset": row["dataset"],
            "budget": row["budget"],
            "seed": row["seed"],
            "method": "full_pool_rsp_auto",
            "test_n": len(gold),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "lambda0": row["lambda0"],
            "selected_uid_sha256": row["expected_selected_uid_sha256"],
            "predictions_file": str(branch / "predictions.npz"),
            "predictions_sha256": sha256(branch / "predictions.npz"),
        })
    if missing:
        (output / "completion.json").write_text(json.dumps({"status": "partial", "missing": missing}, indent=2) + "\n")
        raise SystemExit(f"Incomplete experiment: {len(missing)} branches missing")
    if len(full_rows) != protocol["expected_active_branches"]:
        raise ValueError("Unexpected completed branch count")

    prefilter_all = read_csv(args.prefilter_results.resolve())
    prefilter = {
        (row["dataset"], int(row["budget"]), int(row["seed"])): row
        for row in prefilter_all if row["method"] == "rsp"
    }
    paired = []
    for full in full_rows:
        key = (full["dataset"], int(full["budget"]), int(full["seed"]))
        if key not in prefilter:
            raise ValueError(f"Missing matched prefilter result: {key}")
        pre = prefilter[key]
        paired.append({
            **full,
            "prefilter_accuracy": float(pre["accuracy"]),
            "prefilter_macro_f1": float(pre["macro_f1"]),
            "delta_accuracy_prefilter_minus_full": float(pre["accuracy"]) - full["accuracy"],
            "delta_macro_f1_prefilter_minus_full": float(pre["macro_f1"]) - full["macro_f1"],
            "prefilter_prediction_source": pre["prediction_source"],
            "prefilter_prediction_sha256": pre["prediction_sha256"],
        })
    write_csv(output / "individual_results.csv", paired)

    summary = []
    for dataset, budget in sorted({(row["dataset"], int(row["budget"])) for row in paired}):
        rows = sorted([row for row in paired if row["dataset"] == dataset and int(row["budget"]) == budget], key=lambda row: int(row["seed"]))
        record = {"dataset": dataset, "budget": budget, "n_seeds": len(rows)}
        for name in ("accuracy", "macro_f1", "prefilter_accuracy", "prefilter_macro_f1", "delta_accuracy_prefilter_minus_full", "delta_macro_f1_prefilter_minus_full"):
            mean, sd = mean_sd([float(row[name]) for row in rows])
            record[name + "_mean"] = mean
            record[name + "_sd"] = sd
        summary.append(record)
    write_csv(output / "summary_results.csv", summary)

    lines = [
        "# Full-pool automatic-lambda0 Full RSP entropy ablation",
        "",
        "Only cells where top-min(4B,N) is smaller than the full pool are reported.",
        "All values are percentages; deltas are entropy-prefiltered minus full-pool.",
        "",
        "| B | Dataset | Entropy-prefiltered Acc/F1 | Full-pool Acc/F1 | Delta Acc/F1 |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in sorted(summary, key=lambda row: (int(row["budget"]), row["dataset"])):
        lines.append(
            f"| {row['budget']} | {row['dataset']} | "
            f"{100*row['prefilter_accuracy_mean']:.2f}/{100*row['prefilter_macro_f1_mean']:.2f} | "
            f"{100*row['accuracy_mean']:.2f}/{100*row['macro_f1_mean']:.2f} | "
            f"{100*row['delta_accuracy_prefilter_minus_full_mean']:+.2f}/{100*row['delta_macro_f1_prefilter_minus_full_mean']:+.2f} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "completion.json").write_text(json.dumps({
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "expected_branches": protocol["expected_active_branches"],
        "completed_branches": len(full_rows),
        "missing": [],
        "prefilter_results": str(args.prefilter_results.resolve()),
        "prefilter_results_sha256": sha256(args.prefilter_results.resolve()),
    }, indent=2) + "\n")
    print((output / "completion.json").read_text())


if __name__ == "__main__":
    main()
