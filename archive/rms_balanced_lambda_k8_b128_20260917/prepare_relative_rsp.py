#!/usr/bin/env python3
"""Freeze Relative-RSP inputs and audit its existing dev-tuned comparator.

CPU-only deterministic analysis of retained artifacts. No model is loaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from relative_rsp_utils import (METHOD, METHOD2, METHOD3, METHOD_RMS_BALANCED,
                                derive_relative_rows, derive_relative2_rows,
                                derive_relative3_rows, derive_rms_balanced_rows,
                                ordered_uid_sha256)

DATASETS = ("afrimedqa", "medexpqa", "medexqa", "pubmedqa", "medmcqa")
SEEDS = (13, 23, 37)
LAMBDAS = dict(zip(DATASETS, (0.25, 448.0, 24.0, 0.0, 56.0)))
TEST_N = dict(zip(DATASETS, (892, 125, 200, 200, 4096)))
EXPECTED_TRAIN = {
    "budget": 128, "max_length": 1024, "train_batch_size": 1,
    "eval_batch_size": 4, "gradient_accumulation_steps": 8,
    "epochs": 2.0, "learning_rate": 1e-4, "lora_rank": 16, "lora_alpha": 32,
    "model_name": "google/medgemma-4b-it",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    # str.splitlines() also splits valid Unicode separators *inside* JSON strings.
    # Match the unchanged training loader: iterate physical newline-delimited records.
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        for value in values:
            f.write(json.dumps(value, allow_nan=False) + "\n")


def copy(source, target):
    source, target = Path(source), Path(target)
    require(not target.exists(), f"Refusing overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    digest = sha(source)
    require(sha(target) == digest, f"Copy hash mismatch: {source}")
    return {"path": str(target), "source_path": str(source), "sha256": digest,
            "bytes": target.stat().st_size}


def norm(q):
    return re.sub(r"\s+", " ", "" if q is None else str(q).strip()).strip().lower()


def metrics(p, g):
    prediction = p.argmax(axis=1)
    f1 = []
    for c in range(p.shape[1]):
        tp = np.sum((prediction == c) & (g == c))
        fp = np.sum((prediction == c) & (g != c))
        fn = np.sum((prediction != c) & (g == c))
        f1.append(float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0)
    return {"accuracy": float(np.mean(prediction == g)), "macro_f1": float(np.mean(f1))}


def verify_cached_components(source, metadata, score_rows, pool_hash, source_mc_passes=8,
                             expected_candidate_indices=None):
    clean_dir = Path(metadata["arguments"]["clean_dir"])
    clean_path = clean_dir / "clean_scores.npz"
    require(sha(clean_path) == metadata["clean_score_sha256"], "Clean cache hash mismatch")
    require(read(clean_dir / "clean_metadata.json")["pool_sha256"] == pool_hash,
            "Historical acquisition pool differs from frozen training pool")
    with np.load(clean_path, allow_pickle=False) as z:
        p, indices = z["probabilities"], z["candidate_indices"]
        require(p.ndim == 2 and len(p) == len(score_rows) and np.isfinite(p).all()
                and (p >= 0).all() and (p <= 1).all(), "Invalid clean probability cache")
        np.testing.assert_allclose(p.sum(1), 1, atol=2e-6, rtol=0)
        np.testing.assert_array_equal(z["negative_margin"], [r["margin"] for r in score_rows])
        np.testing.assert_array_equal(z["entropy"], [r["entropy"] for r in score_rows])
        sorted_all = np.sort(p, axis=1)
        np.testing.assert_allclose(sorted_all[:, -1] - sorted_all[:, -2],
                                   -z["negative_margin"], atol=1e-15, rtol=0)
        expected_indices = (np.argsort(-z["entropy"], kind="stable")[:min(512, len(p))]
                            if expected_candidate_indices is None else
                            np.asarray(expected_candidate_indices, dtype=np.int64))
        np.testing.assert_array_equal(indices, expected_indices)
    with np.load(source / "rsp_scores.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(indices, z["candidate_indices"])
        perturbed = z["perturbed_probabilities"]
        require(perturbed.shape == (source_mc_passes, len(indices), p.shape[1]),
                "Wrong cached pass count")
        require(np.isfinite(perturbed).all() and (perturbed >= 0).all() and (perturbed <= 1).all(),
                "Invalid noisy probability cache")
        np.testing.assert_allclose(perturbed.sum(2), 1, atol=2e-6, rtol=0)
        clean = p[indices]
        flipped = perturbed.argmax(2) != clean.argmax(1)[None, :]
        sorted_clean, sorted_noisy = np.sort(clean, axis=1), np.sort(perturbed, axis=2)
        m0 = sorted_clean[:, -1] - sorted_clean[:, -2]
        mk = sorted_noisy[:, :, -1] - sorted_noisy[:, :, -2]
        f = flipped.mean(0)
        d = (np.maximum(0, m0[None, :] - mk) * ~flipped).mean(0)
        np.testing.assert_array_equal(f, z["flip"])
        np.testing.assert_allclose(d, z["drop"], atol=1e-15, rtol=0)
        np.testing.assert_array_equal(f, [score_rows[i]["rsp_flip"] for i in indices])
        np.testing.assert_allclose(d, [score_rows[i]["rsp_drop"] for i in indices], atol=1e-15, rtol=0)
    return clean_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument("--canonical_csv", type=Path, required=True)
    parser.add_argument("--score_variant", choices=("relative1", "relative2", "relative3", "rms_balanced"), default="relative1")
    parser.add_argument("--noise_layer", type=int, default=-1)
    parser.add_argument("--source_mc_passes", type=int, default=8)
    parser.add_argument("--used_mc_passes", type=int, default=8)
    parser.add_argument("--source_scores_root", type=Path)
    parser.add_argument("--source_leaf", default="")
    parser.add_argument("--prefilter_type", choices=("answer_entropy", "aslec_drop", "none"),
                        default="answer_entropy")
    parser.add_argument("--prefilter_scores_root", type=Path)
    parser.add_argument("--method_name")
    args = parser.parse_args()
    method = args.method_name or {
        "relative1": METHOD,
        "relative2": METHOD2,
        "relative3": METHOD3,
        "rms_balanced": METHOD_RMS_BALANCED,
    }[args.score_variant]
    if args.prefilter_type == "aslec_drop":
        require(args.prefilter_scores_root is not None,
                "ASLEC-DROP prefilter requires --prefilter_scores_root")
    run, project = args.run_root.resolve(), args.project_root.resolve()
    require(Path(__file__).resolve() == run / "source/prepare_relative_rsp.py",
            "Execute the staged source snapshot, not an external mutable script")
    require(not (run / "protocol.json").exists(), "Already prepared; refusing to replace a freeze")
    require((run / "source").is_dir(), "Stage immutable source snapshot first")
    run.mkdir(parents=True, exist_ok=True)
    expanded = project / "runs/original_rsp_raw_five_lambda_expanded_dev_20260902"
    prior = project / "runs/original_rsp_raw_tuned_20260902"
    five = project / "runs/original_rsp_raw_five_baselines_20260902"
    selected_manifest = rows(expanded / "selected_test_manifest.jsonl")
    baseline_map = {(r["dataset"], int(r["seed"])): r for r in selected_manifest}
    require(set(baseline_map) == {(d, s) for d in DATASETS for s in SEEDS}, "Baseline coverage mismatch")
    selected = read(expanded / "selected_lambda.json")["selections"]
    require({d: float(selected[d]["lambda"]) for d in DATASETS} == LAMBDAS,
            "Comparator lambda freeze differs from current manuscript")
    canonical = copy(args.canonical_csv, run / "canonical_expected_metrics.csv")
    provenance = {"selected_lambda": copy(expanded / "selected_lambda.json", run / "baseline/selected_lambda.json"),
                  "selected_test_manifest": copy(expanded / "selected_test_manifest.jsonl", run / "baseline/selected_test_manifest.jsonl"),
                  "expanded_protocol": copy(expanded / "protocol.json", run / "baseline/expanded_protocol.json")}
    data_manifest, evaluation, all_data = {}, {}, {}
    for dataset in DATASETS:
        split_hashes, dataset_rows = {}, {}
        for split in ("pool", "dev", "test"):
            split_hashes[split] = copy(project / "data/additional_medical" / dataset / f"{split}.jsonl",
                                       run / "data" / dataset / f"{split}.jsonl")
            dataset_rows[split] = rows(split_hashes[split]["path"])
            split_hashes[split]["rows"] = len(dataset_rows[split])
            split_hashes[split]["ordered_uid_sha256"] = ordered_uid_sha256(r["uid"] for r in dataset_rows[split])
        all_data[dataset] = dataset_rows
        data_manifest[dataset] = {"split_hashes": split_hashes}
        test = dataset_rows["test"]
        require(len(test) == TEST_N[dataset], f"Unexpected test cohort: {dataset}")
        pool_keys = {norm(r["question"]) for r in dataset_rows["pool"]}
        keep = [i for i, r in enumerate(test) if dataset != "medexpqa" or norm(r["question"]) not in pool_keys]
        if dataset == "medexpqa":
            require(len(keep) == 121, "MedExp primary must exactly reproduce 121-case manuscript cohort")
            require([i for i in range(len(test)) if i not in set(keep)] == [4, 25, 41, 84],
                    "MedExp exclusions must match the existing manuscript's exact four indices")
        evaluation[dataset] = {"test_uids": [r["uid"] for r in test],
            "test_gold": [ord(r["answer_letter"]) - ord("A") for r in test],
            "primary_keep_indices": keep,
            "excluded_indices": [i for i in range(len(test)) if i not in set(keep)],
            "primary_condition": "exact_normalized_question_pool_overlap_removed" if dataset == "medexpqa" else "unchanged_test",
            "normalization": "collapse whitespace, strip, lowercase; compare question only to full pool; not label-dependent",
            "raw_n": len(test), "primary_n": len(keep)}
    write(run / "evaluation_manifest.json", evaluation)
    training_rows, baseline_values, score_audits = [], {}, []
    for dataset in DATASETS:
        expected_candidate_indices = None
        if args.prefilter_type == "aslec_drop":
            prefilter_dir = args.prefilter_scores_root.resolve() / dataset
            prefilter_rows = rows(prefilter_dir / "scores.jsonl")
            pool = all_data[dataset]["pool"]
            require([r["uid"] for r in prefilter_rows] == [r["uid"] for r in pool],
                    f"ASLEC score/pool UID order mismatch: {dataset}")
            aslec_values = np.asarray([float(r["aslec_drop"]) for r in prefilter_rows])
            require(np.isfinite(aslec_values).all(), f"Non-finite ASLEC-DROP score: {dataset}")
            expected_candidate_indices = np.argsort(
                -aslec_values, kind="stable")[:min(512, len(pool))]
        elif args.prefilter_type == "none":
            expected_candidate_indices = np.arange(len(all_data[dataset]["pool"]), dtype=np.int64)
        for seed in SEEDS:
            if args.source_scores_root is None:
                source = (five / "rsp_scores" if dataset == "medmcqa" else prior / "scores") / dataset / f"seed_{seed}"
            else:
                source = args.source_scores_root.resolve() / dataset / f"seed_{seed}"
                if args.source_leaf:
                    source /= args.source_leaf
            metadata = read(source / "score_metadata.json")
            require(metadata["dataset"] == dataset and metadata["mc_passes"] == args.source_mc_passes and
                    metadata["noise_layer_offset"] == args.noise_layer and metadata["budget_items"] == 128,
                    f"Scoring protocol mismatch: {dataset}/{seed}")
            require(metadata["adapter_loaded"] is False and
                    metadata["answer_labels_read_during_acquisition"] is False and
                    metadata["rationales_read_during_acquisition"] is False, "Selector visibility mismatch")
            if args.prefilter_type == "aslec_drop":
                require(metadata.get("prefilter_name") == "aslec_drop",
                        f"Scoring prefilter mismatch: {dataset}/{seed}")
            elif args.prefilter_type == "none":
                require(metadata.get("prefilter_name") == "none",
                        f"Scoring no-prefilter protocol mismatch: {dataset}/{seed}")
            require(metadata["arguments"]["noise_sigma"] == .02 and
                    metadata["arguments"]["prefilter_mult"] == 4 and
                    metadata["arguments"]["seed"] == seed, "Noise/prefilter/seed mismatch")
            require(metadata["backbone"] == EXPECTED_TRAIN["model_name"] and
                    metadata["arguments"]["model_name"] == EXPECTED_TRAIN["model_name"],
                    "Scoring backbone identifier mismatch")
            require(sha(source / "scores.jsonl") == metadata["score_file_sha256"], "Source score hash mismatch")
            score_rows = rows(source / "scores.jsonl")
            pool = all_data[dataset]["pool"]
            require([r["uid"] for r in score_rows] == [r["uid"] for r in pool], "Score/pool UID order mismatch")
            clean_dir = verify_cached_components(
                source, metadata, score_rows,
                data_manifest[dataset]["split_hashes"]["pool"]["sha256"],
                source_mc_passes=args.source_mc_passes,
                expected_candidate_indices=expected_candidate_indices)
            score_root = run / "scores" / dataset / f"seed_{seed}"
            source_files = {}
            for name in ("scores.jsonl", "score_metadata.json", "rsp_scores.npz"):
                source_files[name] = copy(source / name, score_root / "retained" / name)
            for name in ("clean_scores.npz", "clean_metadata.json"):
                source_files[name] = copy(clean_dir / name, score_root / "retained" / name)
            if args.score_variant == "rms_balanced":
                derived, diagnostics = derive_rms_balanced_rows(
                    score_rows, method=method,
                    expected_candidate_indices=expected_candidate_indices,
                    prefilter_name=args.prefilter_type)
            elif (args.score_variant in ("relative2", "relative3") or
                    args.source_mc_passes != args.used_mc_passes or args.noise_layer != -1):
                with np.load(clean_dir / "clean_scores.npz", allow_pickle=False) as clean_cache, \
                        np.load(source / "rsp_scores.npz", allow_pickle=False) as noisy_cache:
                    perturbed = noisy_cache["perturbed_probabilities"][:args.used_mc_passes]
                    require(len(perturbed) == args.used_mc_passes,
                            f"Insufficient cached passes: {dataset}/{seed}")
                    if args.score_variant == "relative3":
                        derive = derive_relative3_rows
                        derived, diagnostics = derive(
                            score_rows, clean_cache["probabilities"], noisy_cache["candidate_indices"],
                            perturbed)
                    else:
                        upper_bound = 2.0 if args.score_variant == "relative2" else 1.0
                        derived, diagnostics = derive_relative2_rows(
                            score_rows, clean_cache["probabilities"], noisy_cache["candidate_indices"],
                            perturbed, method=method, upper_bound=upper_bound)
            else:
                derived, diagnostics = derive_relative_rows(
                    score_rows, method=method,
                    expected_candidate_indices=expected_candidate_indices,
                    prefilter_name=args.prefilter_type)
            score_path = score_root / f"scores_{method}.jsonl"
            write_rows(score_path, derived)
            base = baseline_map[(dataset, seed)]
            training = read(base["training_result"])
            require(training["dataset"] == dataset and training["seed"] == seed and
                    training["starting_adapter"] is None and training["budget_items"] == 128,
                    "Baseline training identity mismatch")
            for key, expected in EXPECTED_TRAIN.items():
                require(training["arguments"][key] == expected, f"Baseline training mismatch {dataset}/{seed}/{key}")
            require(sha(training["score_file"]) == training["score_file_sha256"], "Historical training-score hash mismatch")
            comparator_source = ((five / "rsp_scores" if dataset == "medmcqa" else prior / "scores") /
                                 dataset / f"seed_{seed}")
            comparator_rows = rows(comparator_source / "scores.jsonl")
            require([r["uid"] for r in comparator_rows] == [r["uid"] for r in pool],
                    "Comparator score/pool UID order mismatch")
            old_scores = [float(r["rsp_flip"]) + LAMBDAS[dataset] * float(r["rsp_drop"])
                          if r["rsp_flip"] is not None else -1e30 for r in comparator_rows]
            old_order = sorted(range(len(comparator_rows)), key=lambda i: (-old_scores[i], i))[:128]
            old_uids = [comparator_rows[i]["uid"] for i in old_order]
            require(old_uids == training["selected_uid_order"], "Cannot reproduce historical original-RSP selection order")
            require(ordered_uid_sha256(old_uids) == training["selected_uid_sha256"], "Old selection hash mismatch")
            base_root = run / "baseline" / dataset / f"seed_{seed}"
            base_files = {
                "training_result": copy(base["training_result"], base_root / "training_result.json"),
                "comparator_scores": copy(comparator_source / "scores.jsonl",
                                           base_root / "comparator_scores.jsonl"),
                "comparator_score_metadata": copy(comparator_source / "score_metadata.json",
                                                   base_root / "comparator_score_metadata.json"),
            }
            for name in ("result.json", "predictions.npz"):
                base_files[name] = copy(Path(base["output_dir"]) / name, base_root / name)
            evaluation_result = read(base_root / "result.json")
            require(evaluation_result["training_result_sha256"] == base_files["training_result"]["sha256"], "Baseline evaluation/training hash mismatch")
            require(evaluation_result["lambda"] == LAMBDAS[dataset], "Wrong evaluated lambda")
            with np.load(base_root / "predictions.npz", allow_pickle=False) as z:
                p, g = z["test_probs"], z["test_gold"]
                np.testing.assert_array_equal(g, evaluation[dataset]["test_gold"])
                require(p.ndim == 2 and p.shape[0] == len(g) and np.isfinite(p).all(), "Invalid baseline probabilities")
                np.testing.assert_allclose(p.sum(1), 1, atol=2e-6, rtol=0)
                raw = metrics(p, g)
                for metric, value in raw.items():
                    require(abs(value - evaluation_result["test"][metric]) < 1e-12, f"Unreproducible stored baseline {metric}")
                keep = evaluation[dataset]["primary_keep_indices"]
                primary = metrics(p[keep], g[keep])
            baseline_values[(dataset, seed)] = primary
            shared = len(set(old_uids) & set(diagnostics["selected_uids"]))
            diagnostics.update({"dataset": dataset, "seed": seed, "source_files": source_files,
                "source_components_independently_recomputed": True,
                "baseline_lambda": LAMBDAS[dataset], "baseline_selected_uid_sha256": training["selected_uid_sha256"],
                "baseline_selection_overlap_count": shared,
                "baseline_selection_jaccard": shared / (256 - shared),
                "baseline_primary": primary, "baseline_raw_legacy": raw})
            write(score_root / "diagnostics.json", diagnostics)
            diagnostics_path = score_root / "diagnostics.json"
            margin_key = "clean_margin" if "clean_margin" in diagnostics else "raw_margin"
            audit_row = {"dataset": dataset, "seed": seed, "selection_overlap_count": shared,
                         "selected_uid_sha256": diagnostics["selected_uid_sha256"],
                         "near_zero_margin_candidates": diagnostics[margin_key]["near_zero_rows"]}
            if "upper_bound" in diagnostics:
                audit_row.update({
                    "upper_bound": diagnostics["upper_bound"],
                    "candidate_saturation_at_upper_bound": diagnostics["candidate_saturation_at_upper_bound"],
                    "selected_saturation_at_upper_bound": diagnostics["selected_saturation_at_upper_bound"]})
            score_audits.append(audit_row)
            training_rows.append({"dataset": dataset, "seed": seed, "method": method,
                "budget": 128, "layer": args.noise_layer, "passes": args.used_mc_passes,
                "data_dir": str(run / "data" / dataset), "score_file": str(score_path),
                "score_file_sha256": sha(score_path), "expected_selected_uid_sha256": diagnostics["selected_uid_sha256"],
                "score_diagnostics_file": str(diagnostics_path),
                "score_diagnostics_sha256": sha(diagnostics_path),
                "output_dir": str(run / "branches" / dataset / f"seed_{seed}" / method),
                "baseline_prediction_file": str(base_root / "predictions.npz"),
                "baseline_result_file": str(base_root / "result.json"),
                "baseline_training_result_file": str(base_root / "training_result.json"),
                "expected_baseline_selected_uid_sha256": training["selected_uid_sha256"],
                "baseline_lambda": LAMBDAS[dataset], "baseline_method": base["method"],
                "baseline_file_hashes": base_files})
    with Path(canonical["path"]).open() as f:
        expected = {r["dataset"]: r for r in csv.DictReader(f) if r["method"] == "rsp"}
    reproduced = []
    for dataset in (*DATASETS, "mean_across_datasets"):
        row = {"dataset": dataset}
        for metric in ("accuracy", "macro_f1"):
            values = [baseline_values[(dataset, s)][metric] if dataset in DATASETS else
                      float(np.mean([baseline_values[(d, s)][metric] for d in DATASETS])) for s in SEEDS]
            mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
            require(abs(mean - float(expected[dataset][metric + "_mean"])) < 1e-12 and
                    abs(sd - float(expected[dataset][metric + "_sd"])) < 1e-12,
                    f"FAIL: cannot reproduce manuscript RSP {dataset}/{metric}")
            row.update({metric + "_mean": mean, metric + "_sd": sd})
        reproduced.append(row)
    cache = Path("model_cache/models--google--medgemma-4b-it")
    revision = (cache / "refs/main").read_text().strip()
    snapshot = cache / "snapshots" / revision
    model_files = [{"name": p.name, "bytes": p.stat().st_size, "resolved_blob": str(p.resolve()),
                    "sha256": sha(p) if p.suffix in (".json", ".model") else None} for p in sorted(snapshot.iterdir()) if p.is_file()]
    formula = ({"relative1": "mean_k clip(1-signed_margin_k/max(clean_margin,1e-8),0,1)",
                "relative2": "mean_k clip(1-signed_margin_k/max(clean_margin,1e-8),0,2)",
                "relative3": "mean_k clip(1-signed_margin_k/max(clean_margin,1e-8),0,3)",
                "rms_balanced": "F + lambda0*D; lambda0=RMS_candidate(F)/(RMS_candidate(D)+1e-8)"}
               [args.score_variant])
    protocol = {"name": "Relative-RSP fixed formula versus original expanded dev-tuned RSP",
        "created_utc": datetime.now(timezone.utc).isoformat(), "command": shlex.join([sys.executable, *sys.argv]),
        "method": method, "score_variant": args.score_variant, "formula": formula,
        "post_hoc_status": ("test-informed exploratory layer-depth variant"
                             if args.noise_layer != -1 else
                             ("test-informed exploratory upper-bound variant"
                              if args.score_variant == "relative3" else "prespecified for its run")),
        "epsilon": 1e-8, "lambda_search": False, "datasets": DATASETS, "seeds": SEEDS,
        "budget_items": 128, "prefilter_multiplier": (None if args.prefilter_type == "none" else 4),
        "prefilter_type": args.prefilter_type,
        "prefilter_scores_root": (str(args.prefilter_scores_root.resolve())
                                  if args.prefilter_scores_root else None),
        "noise_layer": args.noise_layer, "mc_passes": args.used_mc_passes,
        "source_mc_passes": args.source_mc_passes,
        "source_scores_root": str(args.source_scores_root.resolve()) if args.source_scores_root else None,
        "source_leaf": args.source_leaf or None,
        "cached_prefix_reuse": args.source_mc_passes != args.used_mc_passes,
        "cached_prefix_equivalence_check": (
            "For every target dataset and seed, layer=-1 K=12 prefixes were compared with independent "
            "K=8 caches: candidate indices were identical and max absolute probability difference was 0."
            if args.source_mc_passes != args.used_mc_passes else None),
        "noise_sigma_absolute": .02,
        "noise_location": f"final prompt position at decoder block offset {args.noise_layer}",
        "backbone": "google/medgemma-4b-it", "backbone_revision": revision,
        "model_ref": {"path": str(cache / "refs/main"), "sha256": sha(cache / "refs/main")},
        "model_files": model_files, "starting_adapter": None,
        "training_parameters": EXPECTED_TRAIN, "skip_adapter_save": True,
        "rationale_role": "input_context_only; final answer loss only",
        "selector_visibility": (
            "rationale-aware ASLEC-DROP prefilter followed by question/options-only RSP reranking; "
            "no answer label used in acquisition"
            if args.prefilter_type == "aslec_drop" else
            ("question/options only; no rationale or label used; every pool item receives RSP scoring and is eligible"
             if args.prefilter_type == "none" else
             "question/options only; no rationale or label used in scoring or selection")),
        "development_use": "new method evaluated once per seed, never used to choose settings or checkpoint",
        "comparison": "existing expanded dev-selected lambda results; no baseline retraining",
        "baseline_lambdas": LAMBDAS, "canonical_expected_metrics": canonical["path"],
        "canonical_expected_metrics_sha256": canonical["sha256"], "baseline_provenance": provenance,
        "data_manifest": data_manifest, "evaluation_manifest_sha256": sha(run / "evaluation_manifest.json"),
        "source_snapshot_hashes": {p.name: sha(p) for p in sorted((run / "source").iterdir()) if p.is_file()},
        "git_commit": None, "git_note": "Experiment source directory has no git repository; immutable source SHA256 snapshot used.",
        "limitations": ["Historical acquisition metadata records model ID but not historical weight hashes; current cached revision is recorded, not asserted to prove historical weight identity.",
                        "Three paired acquisition/downstream seeds; not independent acquisition-by-training factorial design.",
                        "128 rationale-containing examples is not equal rationale-token cost.",
                        "Original method has additional dev lambda-grid fit cost; relative method has one fit per dataset/seed.",
                        "Previously viewed tests: descriptive comparison, not fresh confirmatory evidence.",
                        "No matched no-rationale control; improvements cannot identify a causal rationale benefit."]}
    write_rows(run / "training_manifest.jsonl", training_rows)
    protocol["training_manifest_sha256"] = sha(run / "training_manifest.jsonl")
    write(run / "protocol.json", protocol)
    write(run / "preflight.json", {"canonical_verified": True, "new_training_runs": len(training_rows),
        "new_model_inference_performed_during_preflight": False, "original_metrics": reproduced,
        "score_audits": score_audits, "protocol_sha256": sha(run / "protocol.json"),
        "command": protocol["command"], "date": protocol["created_utc"]})
    print(json.dumps({"canonical_verified": True, "new_training_runs": len(training_rows),
                      "run_root": str(run), "original_metrics": reproduced, "score_audits": score_audits}, indent=2))


if __name__ == "__main__":
    main()
