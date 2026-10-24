#!/usr/bin/env python3
"""Freeze a protocol-matched full-pool automatic-lambda0 RSP ablation.

This script reuses the complete current batch-4 K=8/sigma=0.02 Flip and Drop
components retained by the corrected full-pool Drop-only run.  It performs no
model inference.  The automatic coefficient is recomputed over the complete
pool independently for every dataset/seed.  Only dataset-budget cells where
top-min(4B,N) is a strict subset of the full pool are scheduled for training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DATASETS = ("afrimedqa", "medexpqa", "medexqa", "pubmedqa", "medmcqa")
SEEDS = (13, 23, 37)
BUDGETS = (64, 128, 256)
EPSILON = 1e-8
METHOD = "full_pool_rsp_auto_k8"


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


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def uid_hash(pool: list[dict], indices: list[int]) -> str:
    text = "\n".join(pool[index]["uid"] for index in indices)
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument("--score_source", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_root.resolve()
    project = args.project_root.resolve()
    source = args.score_source.resolve()
    if (run / "protocol.json").exists():
        raise ValueError("Refusing to overwrite a frozen run")

    source_protocol = read_json(source / "protocol.json")
    required = {
        "method": "full_pool_drop_only_current",
        "noise_layer": -1,
        "mc_passes": 8,
        "noise_sigma": 0.02,
        "scoring_batch_size": 4,
        "training_order": "ascending original pool index",
        "backbone": "google/medgemma-4b-it",
    }
    for key, expected in required.items():
        if source_protocol.get(key) != expected:
            raise ValueError(f"Incompatible score source {key}: {source_protocol.get(key)!r} != {expected!r}")
    if source_protocol.get("training") != {
        "epochs": 2.0,
        "learning_rate": 0.0001,
        "lora_alpha": 32,
        "lora_rank": 16,
        "max_length": 1024,
    }:
        raise ValueError("Incompatible source training freeze")

    manifest: list[dict] = []
    selections: list[dict] = []
    diagnostics: list[dict] = []
    input_hashes = {str(source / "protocol.json"): sha256(source / "protocol.json")}

    for dataset in DATASETS:
        data_dir = project / "data" / "additional_medical" / dataset
        pool_path = data_dir / "pool.jsonl"
        pool = read_jsonl(pool_path)
        input_hashes[str(pool_path)] = sha256(pool_path)
        for seed in SEEDS:
            score_path = source / "scores" / dataset / f"seed_{seed}" / "full_pool_drop_only_scores.jsonl"
            scores = read_jsonl(score_path)
            input_hashes[str(score_path)] = sha256(score_path)
            if [row["uid"] for row in scores] != [row["uid"] for row in pool]:
                raise ValueError(f"UID/order mismatch: {dataset}/{seed}")
            flip = np.asarray([row["rsp_flip"] for row in scores], dtype=float)
            drop = np.asarray([row["rsp_drop"] for row in scores], dtype=float)
            if not (np.isfinite(flip).all() and np.isfinite(drop).all()):
                raise ValueError(f"Incomplete score components: {dataset}/{seed}")
            if (flip < 0).any() or (drop < 0).any():
                raise ValueError(f"Negative score component: {dataset}/{seed}")
            a_f = float(np.sqrt(np.mean(flip * flip)))
            a_d = float(np.sqrt(np.mean(drop * drop)))
            if a_f <= EPSILON:
                lambda0 = 0.0
                priority = drop.copy()
                score_rule = "drop_only_no_flip_fallback"
                fallback = True
            else:
                lambda0 = a_f / (a_d + EPSILON)
                priority = flip + lambda0 * drop
                score_rule = "full_rsp_automatic_lambda0"
                fallback = False
            for row, value in zip(scores, priority):
                row[METHOD] = float(value)
            derived = run / "scores" / dataset / f"seed_{seed}" / "full_pool_rsp_auto_scores.jsonl"
            write_jsonl(derived, scores)
            input_hashes[str(derived)] = sha256(derived)
            diagnostics.append({
                "dataset": dataset,
                "seed": seed,
                "pool_n": len(pool),
                "a_F": a_f,
                "a_D": a_d,
                "lambda0": lambda0,
                "score_rule": score_rule,
                "fallback_triggered": fallback,
                "flip_positive": int((flip > 0).sum()),
                "flip_unique": int(len(np.unique(flip))),
                "drop_positive": int((drop > 0).sum()),
                "drop_unique": int(len(np.unique(drop))),
                "priority_unique": int(len(np.unique(priority))),
                "source_score": str(score_path),
                "source_score_sha256": sha256(score_path),
                "derived_score": str(derived),
                "derived_score_sha256": sha256(derived),
            })
            for budget in BUDGETS:
                candidate_n = min(4 * budget, len(pool))
                entropy_prefilter_active = candidate_n < len(pool)
                ranked = sorted(range(len(pool)), key=lambda index: (-priority[index], index))
                selected = sorted(ranked[:budget])
                selection = {
                    "dataset": dataset,
                    "seed": seed,
                    "budget": budget,
                    "pool_n": len(pool),
                    "top4b_candidate_n": candidate_n,
                    "entropy_prefilter_active": entropy_prefilter_active,
                    "expected_selected_indices": selected,
                    "expected_selected_uid_sha256": uid_hash(pool, selected),
                    "lambda0": lambda0,
                    "a_F": a_f,
                    "a_D": a_d,
                    "score_rule": score_rule,
                    "fallback_triggered": fallback,
                }
                selections.append(selection)
                if not entropy_prefilter_active:
                    continue
                manifest.append({
                    **selection,
                    "method_label": "full_pool_rsp_auto",
                    "score_method": METHOD,
                    "score_file": str(derived),
                    "score_file_sha256": sha256(derived),
                    "data_dir": str(data_dir),
                    "output_dir": str(run / "branches" / dataset / f"seed_{seed}" / f"b{budget}"),
                })

    if len(selections) != 45 or len(manifest) != 27:
        raise ValueError(f"Unexpected cell counts: {len(selections)} selections, {len(manifest)} active branches")
    write_jsonl(run / "training_manifest.jsonl", manifest)
    write_json(run / "selection_audit.json", selections)
    write_json(run / "score_diagnostics.json", diagnostics)
    protocol = {
        "name": "Protocol-matched full-pool automatic-lambda0 Full RSP entropy ablation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": DATASETS,
        "seeds": SEEDS,
        "budgets": BUDGETS,
        "method": "full_pool_rsp_auto",
        "candidate_rule": "complete pool; no entropy eligibility filter or entropy ranking",
        "reported_cells": "only settings with min(4B,N) < N",
        "score_definition": "F + lambda0*D; lambda0=RMS_pool(F)/(RMS_pool(D)+1e-8); raw D fallback if RMS_pool(F)<=1e-8",
        "score_column": METHOD,
        "selector_visibility": "question and options only; no candidate rationale or answer label",
        "noise_layer": -1,
        "mc_passes": 8,
        "noise_sigma": 0.02,
        "scoring_batch_size": 4,
        "training_order": "ascending original pool index",
        "backbone": "google/medgemma-4b-it",
        "training": source_protocol["training"],
        "score_source": str(source),
        "score_source_protocol_sha256": sha256(source / "protocol.json"),
        "input_sha256": input_hashes,
        "training_manifest_sha256": sha256(run / "training_manifest.jsonl"),
        "selection_audit_sha256": sha256(run / "selection_audit.json"),
        "score_diagnostics_sha256": sha256(run / "score_diagnostics.json"),
        "expected_active_branches": 27,
        "excluded_historical_runs": [
            "no_prefilter_relative_rsp_m1_k8_b128_20260917_recovery (different relative-score formula)",
            "ra_rsp_full_pool_k8_m1_sigma002_b128_20260920 (candidate rationale visible)",
            "full_pool_drop_only_k8_b64_128_256_20260921 (obsolete batch-16 cache)",
        ],
        "source_snapshot_hashes": {
            path.name: sha256(path) for path in sorted((run / "source").glob("*.py"))
        },
    }
    write_json(run / "protocol.json", protocol)
    print(json.dumps({"status": "frozen", "active_branches": len(manifest), "all_cells": len(selections)}, indent=2))


if __name__ == "__main__":
    main()
