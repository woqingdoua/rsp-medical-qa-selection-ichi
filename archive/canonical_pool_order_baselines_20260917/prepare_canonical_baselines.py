#!/usr/bin/env python3
"""Freeze canonical pool-index reruns for all retained paper baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DATASETS = ("afrimedqa", "medexpqa", "medexqa", "pubmedqa", "medmcqa")
SEEDS = (13, 23, 37)
METHODS = (
    "entropy", "entropy_prefiltered_random", "margin", "bald_hidden_dropout",
    "batchbald_hidden_dropout", "noise_stability", "sift", "nait", "rsr",
    "aslec_drop", "temp",
)
SENTINEL = -1e30


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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def canonical_selection(pool: list[dict], scores: list[dict], method: str) -> tuple[list[int], str]:
    if len(pool) != len(scores) or any(a["uid"] != b["uid"] for a, b in zip(pool, scores)):
        raise ValueError("Pool/score identity mismatch")
    ranked = sorted(range(len(pool)), key=lambda index: (-float(scores[index][method]), index))[:128]
    canonical = sorted(ranked)
    return canonical, hashlib.sha256("\n".join(pool[index]["uid"] for index in canonical).encode()).hexdigest()


def rows_by_key(path: Path, permitted: set[str]) -> dict[tuple[str, int, str], dict]:
    result = {}
    for row in read_jsonl(path):
        if row["method"] in permitted:
            key = (row["dataset"], int(row["seed"]), row["method"])
            if key in result:
                raise ValueError(f"Duplicate manifest entry: {key}")
            result[key] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--project_root", type=Path, required=True)
    args = parser.parse_args()
    run, project = args.run_root.resolve(), args.project_root.resolve()
    if Path(__file__).resolve() != run / "source" / Path(__file__).name:
        raise ValueError("Execute the frozen source snapshot")
    if (run / "protocol.json").exists():
        raise ValueError("Run already prepared")
    tuned_root = project / "runs/original_rsp_raw_tuned_20260902"
    extra_root = project / "runs/original_rsp_raw_five_baselines_20260902"
    five_root = project / "runs/latest_five_baselines_20260903"
    direct = rows_by_key(tuned_root / "training_manifest.jsonl", {"entropy", "margin"})
    direct.update(rows_by_key(extra_root / "medmcqa_training_manifest.jsonl", {"entropy", "margin"}))
    extra = rows_by_key(extra_root / "extra_training_manifest.jsonl", {"bald_hidden_dropout", "batchbald_hidden_dropout", "noise_stability", "sift"})
    latest = rows_by_key(five_root / "direct_training_manifest.jsonl", {"nait", "rsr", "aslec_drop"})
    temp = rows_by_key(five_root / "temp_training_manifest.jsonl", {"temp"})
    sources = {**direct, **extra, **latest, **temp}
    expected_source = {(dataset, seed, method) for dataset in DATASETS for seed in SEEDS for method in METHODS if method != "entropy_prefiltered_random"}
    if set(sources) != expected_source:
        missing, extra_keys = expected_source - set(sources), set(sources) - expected_source
        raise ValueError(f"Baseline source coverage mismatch: missing={sorted(missing)}, extra={sorted(extra_keys)}")
    manifest, inputs = [], {}
    for dataset in DATASETS:
        data_dir = project / "data/additional_medical" / dataset
        pool = read_jsonl(data_dir / "pool.jsonl")
        for split in ("pool", "dev", "test"):
            inputs[str(data_dir / f"{split}.jsonl")] = sha256(data_dir / f"{split}.jsonl")
        for seed in SEEDS:
            # The control samples uniformly from the same entropy top-4B pool.
            raw_score = (extra_root / "rsp_scores" / dataset / f"seed_{seed}/scores.jsonl"
                         if dataset == "medmcqa" else tuned_root / f"scores/{dataset}/seed_{seed}/scores.jsonl")
            raw_rows = read_jsonl(raw_score)
            candidates = sorted(range(len(raw_rows)), key=lambda index: (-float(raw_rows[index]["entropy"]), index))[:min(512, len(raw_rows))]
            candidate_set = set(candidates)
            derived = [dict(row, entropy_prefiltered_random=(float(row["random"]) if index in candidate_set else SENTINEL))
                       for index, row in enumerate(raw_rows)]
            epr_path = run / f"derived_scores/entropy_prefiltered_random/{dataset}/seed_{seed}/scores.jsonl"
            write_jsonl(epr_path, derived)
            sources[(dataset, seed, "entropy_prefiltered_random")] = {
                "dataset": dataset, "seed": seed, "method": "entropy_prefiltered_random",
                "data_dir": str(data_dir), "score_file": str(epr_path),
                "source_score_file": str(raw_score), "source_score_sha256": sha256(raw_score),
                "definition": "top min(4B,N) entropy candidates then deterministic stored random score",
            }
            for method in METHODS:
                source = sources[(dataset, seed, method)]
                score_file = Path(source["score_file"])
                scores = read_jsonl(score_file)
                selected, selected_hash = canonical_selection(pool, scores, method)
                inputs[str(score_file)] = sha256(score_file)
                manifest.append({
                    "dataset": dataset, "seed": seed, "method_label": method,
                    "score_method": method, "score_file": str(score_file),
                    "score_file_sha256": inputs[str(score_file)], "data_dir": str(data_dir),
                    "expected_selected_indices": selected, "expected_selected_uid_sha256": selected_hash,
                    "output_dir": str(run / "branches" / dataset / f"seed_{seed}" / method),
                })
    manifest_path = run / "training_manifest.jsonl"
    write_jsonl(manifest_path, manifest)
    protocol = {
        "name": "Canonical pool-index training-order baseline rerun",
        "created_utc": datetime.now(timezone.utc).isoformat(), "methods": METHODS,
        "datasets": DATASETS, "seeds": SEEDS, "budget_items": 128,
        "selection_rule": "retain each baseline's frozen top-128 score set",
        "training_order": "ascending original pool index", "reused_canonical_rsp_run": str(project / "runs/canonical_pool_order_rsp_pair_20260917"),
        "training_manifest_sha256": sha256(manifest_path), "input_sha256": inputs,
        "source_snapshot_hashes": {path.name: sha256(path) for path in sorted((run / "source").iterdir()) if path.is_file()},
    }
    write_json(run / "protocol.json", protocol)
    print(json.dumps({"run_root": str(run), "branches": len(manifest), "methods": METHODS}, indent=2))


if __name__ == "__main__":
    main()
