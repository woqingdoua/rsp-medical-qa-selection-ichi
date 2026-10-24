#!/usr/bin/env python
"""Run the repository's recorded TEMP adaptation from raw MedGemma."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from common import load_model, load_tokenizer, read_jsonl, set_seed, write_json, write_jsonl
from original_rsp_raw_utils import sha256
import rationale_baselines as rb


CALIBRATION_HISTORY: dict[str, list[dict[str, float]]] = {}


def _bracketed_search(evaluate, initial: float, low_target: float, high_target: float,
                      max_steps: int, label: str) -> tuple[float, float]:
    """Find a perturbation scale in a target ratio interval without overshoot bias."""
    if max_steps < 1:
        raise ValueError("calibration_steps must be positive")
    midpoint = 0.5 * (low_target + high_target)
    low: tuple[float, float] | None = None
    high: tuple[float, float] | None = None
    scale = float(initial)
    history: list[dict[str, float]] = []
    for _ in range(max_steps):
        ratio = float(evaluate(scale))
        history.append({"scale": scale, "ratio": ratio})
        print(f"[temp] {label}={scale:.6g}, calibration loss ratio={ratio:.3f}", flush=True)
        if low_target <= ratio <= high_target:
            CALIBRATION_HISTORY[label] = history
            return scale, ratio
        if ratio < low_target:
            low = (scale, ratio)
            scale = scale * 1.8 if high is None else float((scale * high[0]) ** 0.5)
        else:
            high = (scale, ratio)
            scale = scale * 0.65 if low is None else float((low[0] * scale) ** 0.5)
    best = min(history, key=lambda item: abs(np.log(max(item["ratio"], 1e-12) / midpoint)))
    CALIBRATION_HISTORY[label] = history
    return float(best["scale"]), float(best["ratio"])


def _calibrate_random_noise(model, tokenizer, rows, noise_targets, args, clean):
    def evaluate(sigma: float) -> float:
        rb._apply_relative_weight_noise(noise_targets, sigma, args.seed + 7001, +1.0)
        try:
            noisy = rb._summed_prefix_losses(
                model, tokenizer, rows, args.max_length, args.temp_difficulty_tokens
            )
        finally:
            rb._apply_relative_weight_noise(noise_targets, sigma, args.seed + 7001, -1.0)
        return float(np.nanmean(noisy) / max(float(np.nanmean(clean)), 1e-8))

    return _bracketed_search(
        evaluate,
        args.temp_random_sigma,
        args.temp_random_target_low,
        args.temp_random_target_high,
        args.temp_calibration_steps,
        "random_sigma",
    )


def _calibrate_direction_scale(model, tokenizer, rows, triplets, args, clean):
    def evaluate(scale: float) -> float:
        rb._set_adapter_state(triplets, scale, args.seed + 9001)
        perturbed = rb._summed_prefix_losses(
            model, tokenizer, rows, args.max_length, args.temp_reasoning_tokens
        )
        return float(np.nanmean(perturbed) / max(float(np.nanmean(clean)), 1e-8))

    try:
        return _bracketed_search(
            evaluate,
            args.temp_direction_scale,
            2.5,
            3.5,
            args.temp_calibration_steps,
            "direction_scale",
        )
    finally:
        rb._set_adapter_state(triplets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--direction_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="google/medgemma-4b-it")
    parser.add_argument("--cache_dir", default="model_cache")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--difficulty_tokens", type=int, default=100)
    parser.add_argument("--reasoning_tokens", type=int, default=384)
    parser.add_argument("--checkpoints", type=int, default=4)
    parser.add_argument("--per_cluster", type=int, default=4)
    parser.add_argument("--calibration_samples", type=int, default=8)
    parser.add_argument("--calibration_steps", type=int, default=6)
    parser.add_argument("--random_sigma", type=float, default=0.02)
    parser.add_argument("--direction_scale", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--pool_limit", type=int, default=0, help="Prefix limit for smoke tests only")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    direction_weights = args.direction_path / "adapter_model.safetensors"
    if not direction_weights.exists():
        raise FileNotFoundError(direction_weights)
    set_seed(args.seed)
    started = time.time()
    pool_file = args.data_dir / "pool.jsonl"
    rows = read_jsonl(pool_file)
    if args.pool_limit:
        rows = rows[: args.pool_limit]
    tokenizer = load_tokenizer(args.model_name, args.cache_dir)
    # Fresh LoRA reproduces theta_0 for the direction adapter trained with the same seed.
    model = load_model(
        args.model_name,
        args.cache_dir,
        adapter_path=None,
        trainable=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    temp_args = SimpleNamespace(
        temp_direction_path=str(args.direction_path),
        seed=args.seed,
        max_length=args.max_length,
        temp_difficulty_tokens=args.difficulty_tokens,
        temp_reasoning_tokens=args.reasoning_tokens,
        temp_checkpoints=args.checkpoints,
        temp_per_cluster=args.per_cluster,
        temp_calibration_samples=args.calibration_samples,
        temp_calibration_steps=args.calibration_steps,
        temp_random_sigma=args.random_sigma,
        temp_random_target_low=2.0,
        temp_random_target_high=3.0,
        temp_direction_scale=args.direction_scale,
        temp_budget=args.budget,
        log_every=args.log_every,
    )
    # Patch only this executable's TEMP call; the historical implementation and
    # its completed artifacts remain untouched for provenance.
    rb._calibrate_random_noise = _calibrate_random_noise
    rb._calibrate_direction_scale = _calibrate_direction_scale
    scores, temp_metadata = rb.score_temp(model, tokenizer, rows, temp_args)
    for index, row in enumerate(scores):
        row["index"] = index
        row["selection_unit_cost"] = 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "scores.jsonl"
    write_jsonl(score_path, scores)
    metadata = {
        "protocol": "latest raw-MedGemma five-dataset baseline extension 2026-09-03",
        "dataset": args.dataset,
        "method": "temp",
        "method_name": "Token-Efficient Model Perturbation",
        "adaptation": "repository PEFT/MedGemma adaptation with a dev-rationale auxiliary LoRA direction",
        "source_reference": "Jin et al., Reasoning Quality Emerges Early, arXiv:2606.26797",
        "backbone": args.model_name,
        "answer_only_warmup": False,
        "scoring_initialization": "fresh raw-backbone LoRA theta_0",
        "pool_selector_visible_fields": ["uid", "question", "options", "domain", "rationale"],
        "pool_rationales_used": True,
        "pool_answers_used": False,
        "visibility_note": "privileged rationale-visible comparator required by TEMP trace losses",
        "direction_path": str(args.direction_path),
        "direction_weights_sha256": sha256(direction_weights),
        "pool_file": str(pool_file),
        "pool_sha256": sha256(pool_file),
        "rows": len(rows),
        "score_file": str(score_path),
        "score_file_sha256": sha256(score_path),
        "selection_direction": "larger is selected earlier",
        "calibration_policy": {
            "algorithm": "geometric bracketing followed by geometric bisection; retain closest tried point if the interval is not reached",
            "random_target_ratio": [2.0, 3.0],
            "direction_target_ratio": [2.5, 3.5],
            "history": CALIBRATION_HISTORY,
        },
        "temp": temp_metadata,
        "elapsed_seconds": time.time() - started,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    write_json(args.output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
