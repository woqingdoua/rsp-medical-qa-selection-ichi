#!/usr/bin/env python
"""Score Original RSP on raw MedGemma, with no answer-only warm-up."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import get_transformer_blocks, load_model, load_tokenizer, read_jsonl, set_seed, write_json, write_jsonl
from original_rsp_raw_utils import (
    AbsoluteGaussianHook,
    entropy_rows,
    margin_rows,
    next_letter_probabilities,
    rsp_components,
    sanitize_for_acquisition,
    sha256,
)


def clean_phase(args, rows: list[dict], visible: list[dict]) -> None:
    tokenizer = load_tokenizer(args.model_name, args.cache_dir)
    model = load_model(args.model_name, args.cache_dir, None, trainable=False)
    probabilities = next_letter_probabilities(
        model, tokenizer, visible, args.batch_size, args.max_length
    )
    entropy = entropy_rows(probabilities)
    margin = margin_rows(probabilities)
    if args.prefilter_name == "none":
        # This ablation deliberately scores every pool item.  Entropy is still
        # retained as a diagnostic, but it must not affect eligibility.
        candidate_indices = np.arange(len(rows), dtype=np.int64)
        candidate_count = len(candidate_indices)
        phase_description = "no candidate prefilter"
    else:
        candidate_count = min(len(rows), max(args.budget, args.prefilter_mult * args.budget))
        candidate_indices = np.argsort(-entropy, kind="stable")[:candidate_count]
        phase_description = "clean entropy prefilter"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "clean_scores.npz",
        probabilities=probabilities,
        entropy=entropy,
        negative_margin=-margin,
        candidate_indices=candidate_indices,
    )
    write_json(
        args.output_dir / "clean_metadata.json",
        {
            "protocol": "Original RSP--Raw",
            "phase": phase_description,
            "dataset": args.dataset,
            "backbone": args.model_name,
            "adapter_loaded": False,
            "answer_labels_read": False,
            "rationales_read": False,
            "pool_rows": len(rows),
            "budget_items": args.budget,
            "entropy_prefilter_multiplier": args.prefilter_mult,
            "entropy_prefilter_rows": candidate_count,
            "pool_sha256": sha256(args.data_dir / "pool.jsonl"),
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
    )


def rsp_phase(args, rows: list[dict], visible: list[dict]) -> None:
    clean_path = args.clean_dir / "clean_scores.npz"
    clean = np.load(clean_path)
    probabilities = clean["probabilities"]
    entropy = clean["entropy"]
    negative_margin = clean["negative_margin"]
    candidate_indices = clean["candidate_indices"].astype(np.int64)
    candidates = [visible[index] for index in candidate_indices]

    tokenizer = load_tokenizer(args.model_name, args.cache_dir)
    model = load_model(args.model_name, args.cache_dir, None, trainable=False)
    blocks = get_transformer_blocks(model)
    layer_index = args.noise_layer if args.noise_layer >= 0 else len(blocks) + args.noise_layer
    if not 0 <= layer_index < len(blocks):
        raise ValueError(f"Invalid noise layer {args.noise_layer} for {len(blocks)} blocks")
    noisy = []
    for pass_index in range(args.mc_passes):
        hook = AbsoluteGaussianHook(
            args.noise_sigma,
            args.seed * 1_000_003 + pass_index * 104729,
        )
        handle = blocks[layer_index].register_forward_hook(hook)
        try:
            noisy.append(
                next_letter_probabilities(
                    model, tokenizer, candidates, args.batch_size, args.max_length
                )
            )
        finally:
            handle.remove()
        print(f"[original-rsp] pass {pass_index + 1}/{args.mc_passes}", flush=True)
    perturbed = np.stack(noisy, axis=0)
    components = rsp_components(
        probabilities[candidate_indices], perturbed, args.drop_weight
    )
    sentinel = -1e30
    rsp = np.full(len(rows), sentinel, dtype=np.float64)
    flip = np.full(len(rows), np.nan, dtype=np.float64)
    drop = np.full(len(rows), np.nan, dtype=np.float64)
    rsp[candidate_indices] = components["rsp"]
    flip[candidate_indices] = components["flip"]
    drop[candidate_indices] = components["drop"]
    random_score = np.random.default_rng(args.seed).random(len(rows))
    eligible = set(candidate_indices.tolist())
    output_rows = []
    for index, row in enumerate(rows):
        output_rows.append(
            {
                "index": index,
                "uid": row["uid"],
                "rsp": float(rsp[index]),
                "random": float(random_score[index]),
                "entropy": float(entropy[index]),
                "margin": float(negative_margin[index]),
                "rsp_flip": None if np.isnan(flip[index]) else float(flip[index]),
                "rsp_drop": None if np.isnan(drop[index]) else float(drop[index]),
                "rsp_prefilter_eligible": index in eligible,
                "selection_unit_cost": 1,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "scores.jsonl", output_rows)
    np.savez_compressed(
        args.output_dir / "rsp_scores.npz",
        candidate_indices=candidate_indices,
        perturbed_probabilities=perturbed,
        **components,
    )
    metadata = {
        "protocol": "Original RSP--Raw",
        "dataset": args.dataset,
        "backbone": args.model_name,
        "scoring_initialization": "raw_backbone_without_answer_only_warmup",
        "adapter_loaded": False,
        "selector_visible_fields": ["uid", "question", "options", "domain"],
        "answer_labels_read_during_acquisition": False,
        "rationales_read_during_acquisition": False,
        "score_definition": "Flip + drop_weight * non-flip positive top1/top2 probability-margin degradation",
        "mc_passes": args.mc_passes,
        "noise_distribution": f"N(0, {args.noise_sigma}^2 I) absolute hidden-state noise",
        "noise_layer_offset": args.noise_layer,
        "resolved_noise_layer": layer_index,
        "drop_weight": args.drop_weight,
        "prefilter_name": args.prefilter_name,
        "prefilter_rows": len(candidate_indices),
        "entropy_prefilter_rows": len(candidate_indices),
        "budget_items": args.budget,
        "clean_score_sha256": sha256(clean_path),
        "score_file_sha256": sha256(args.output_dir / "scores.jsonl"),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    write_json(args.output_dir / "score_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("clean", "rsp"), required=True)
    parser.add_argument(
        "--dataset",
        choices=(
            "medmcqa",
            "medprompt",
            "afrimedqa",
            "medexpqa",
            "medexqa",
            "pubmedqa",
        ),
        required=True,
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--clean_dir", type=Path)
    parser.add_argument("--model_name", default="google/medgemma-4b-it")
    parser.add_argument("--cache_dir", default="model_cache")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--prefilter_mult", type=int, default=4)
    parser.add_argument("--prefilter_name", choices=("answer_entropy", "aslec_drop", "none"),
                        default="answer_entropy")
    parser.add_argument("--mc_passes", type=int, default=8)
    parser.add_argument("--noise_layer", type=int, required=True)
    parser.add_argument("--noise_sigma", type=float, required=True)
    parser.add_argument("--drop_weight", type=float, default=0.5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.phase == "rsp" and args.clean_dir is None:
        parser.error("--clean_dir is required for --phase rsp")
    set_seed(args.seed)
    started = time.time()
    rows = read_jsonl(args.data_dir / "pool.jsonl")
    visible = [sanitize_for_acquisition(row) for row in rows]
    if args.phase == "clean":
        clean_phase(args, rows, visible)
    else:
        rsp_phase(args, rows, visible)
    print(json.dumps({"phase": args.phase, "elapsed_seconds": time.time() - started}))


if __name__ == "__main__":
    main()
