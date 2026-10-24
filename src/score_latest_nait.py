#!/usr/bin/env python
"""Question-only NAIT adaptation for the frozen raw-MedGemma protocol.

The source NAIT algorithm obtains one oriented PCA direction per decoder layer
from first-to-last-token activation changes on an in-domain reference set, then
ranks candidates by their summed projection onto these directions.  Here both
the reference split and candidate pool are sanitized to question/options only
so the selector remains label- and rationale-blind.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.decomposition import PCA

from common import answer_prompt, get_transformer_blocks, load_model, load_tokenizer, read_jsonl, set_seed, write_json, write_jsonl
from original_rsp_raw_utils import sanitize_for_acquisition, sha256


def _hidden_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported transformer block output: {type(output)!r}")


@torch.no_grad()
def activation_deltas(model, tokenizer, rows: Sequence[dict], batch_size: int,
                      max_length: int) -> list[np.ndarray]:
    """Return [N,H] first-to-last token activation changes for every layer."""
    model.eval()
    blocks = get_transformer_blocks(model)
    collected: list[list[np.ndarray]] = [[] for _ in blocks]
    current_first: torch.Tensor | None = None

    def make_hook(layer_index: int):
        def hook(_module, _inputs, output):
            if current_first is None:
                raise RuntimeError("NAIT token indices were not initialized")
            hidden = _hidden_tensor(output)
            batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
            first = hidden[batch_indices, current_first]
            last = hidden[:, -1]
            collected[layer_index].append((last - first).float().cpu().numpy().astype(np.float32))
        return hook

    handles = [block.register_forward_hook(make_hook(index)) for index, block in enumerate(blocks)]
    try:
        device = next(model.parameters()).device
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = tokenizer(
                [answer_prompt(row) for row in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            current_first = encoded["attention_mask"].argmax(dim=1)
            model(**encoded, use_cache=False, return_dict=True)
            if (start + len(batch)) % max(20, batch_size) == 0 or start + len(batch) == len(rows):
                print(f"[nait-activations] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    return [np.concatenate(parts, axis=0) for parts in collected]


def oriented_first_pc(values: np.ndarray, seed: int) -> np.ndarray:
    """First PCA component, sign-aligned with the mean activation change."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(f"PCA needs at least two rows, got {values.shape}")
    component = PCA(n_components=1, svd_solver="randomized", random_state=seed).fit(values).components_[0]
    mean_difference = values.mean(axis=0)
    if float(mean_difference @ component) < 0.0:
        component = -component
    norm = float(np.linalg.norm(component))
    if norm <= 0.0:
        raise RuntimeError("Degenerate NAIT PCA direction")
    return (component / norm).astype(np.float32)


def nait_scores(reference: Sequence[np.ndarray], candidates: Sequence[np.ndarray], seed: int):
    if len(reference) != len(candidates):
        raise ValueError("Reference/candidate layer count mismatch")
    scores = np.zeros(candidates[0].shape[0], dtype=np.float64)
    directions = []
    explained_variance = []
    for layer_index, (ref, candidate) in enumerate(zip(reference, candidates)):
        direction = oriented_first_pc(ref, seed + layer_index * 1009)
        scores += np.asarray(candidate, dtype=np.float64) @ direction.astype(np.float64)
        directions.append(direction)
        # Auditable descriptive statistic; the ranking itself uses the exact direction above.
        centered = np.asarray(ref, dtype=np.float64) - np.asarray(ref, dtype=np.float64).mean(0)
        projected = centered @ direction.astype(np.float64)
        total_variance = float(np.square(centered).sum())
        explained_variance.append(float(np.square(projected).sum() / max(total_variance, 1e-12)))
    return scores, directions, explained_variance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="google/medgemma-4b-it")
    parser.add_argument("--cache_dir", default="model_cache")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--reference_split", default="dev", choices=("dev",))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--pool_limit", type=int, default=0, help="Prefix limit for smoke tests only")
    parser.add_argument("--reference_limit", type=int, default=0, help="Prefix limit for smoke tests only")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(args.seed)
    started = time.time()
    pool_file = args.data_dir / "pool.jsonl"
    reference_file = args.data_dir / f"{args.reference_split}.jsonl"
    pool = [sanitize_for_acquisition(row) for row in read_jsonl(pool_file)]
    reference_rows = [sanitize_for_acquisition(row) for row in read_jsonl(reference_file)]
    if args.pool_limit:
        pool = pool[: args.pool_limit]
    if args.reference_limit:
        reference_rows = reference_rows[: args.reference_limit]
    tokenizer = load_tokenizer(args.model_name, args.cache_dir)
    model = load_model(args.model_name, args.cache_dir, None, trainable=False)
    reference_activations = activation_deltas(
        model, tokenizer, reference_rows, args.batch_size, args.max_length
    )
    candidate_activations = activation_deltas(
        model, tokenizer, pool, args.batch_size, args.max_length
    )
    scores, directions, explained = nait_scores(reference_activations, candidate_activations, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = [
        {"index": index, "uid": row["uid"], "nait": float(scores[index]), "selection_unit_cost": 1}
        for index, row in enumerate(pool)
    ]
    score_path = args.output_dir / "scores.jsonl"
    write_jsonl(score_path, output_rows)
    np.savez_compressed(
        args.output_dir / "nait_directions.npz",
        **{f"layer_{index}": direction for index, direction in enumerate(directions)},
    )
    metadata = {
        "protocol": "latest raw-MedGemma five-dataset baseline extension 2026-09-03",
        "dataset": args.dataset,
        "method": "nait",
        "method_name": "Neuron-Aware Instruction-Tuning data selection",
        "adaptation": "question/options-only activation deltas to preserve the frozen selector visibility rule",
        "source_algorithm": "oriented first PCA component per layer; summed candidate projections",
        "source_reference": "Chen et al., Neuron-Aware Data Selection in Instruction Tuning for Large Language Models, ICLR 2026 / arXiv:2603.13201",
        "backbone": args.model_name,
        "adapter_loaded": False,
        "scoring_initialization": "raw_backbone_without_answer_only_warmup",
        "selector_visible_fields": ["uid", "question", "options", "domain"],
        "answer_labels_used": False,
        "rationales_used": False,
        "reference_split": args.reference_split,
        "reference_labels_used": False,
        "reference_rationales_used": False,
        "reference_rows": len(reference_rows),
        "pool_rows": len(pool),
        "layers": len(directions),
        "explained_variance_ratio_by_layer": explained,
        "pool_file": str(pool_file),
        "pool_sha256": sha256(pool_file),
        "reference_file": str(reference_file),
        "reference_sha256": sha256(reference_file),
        "score_file": str(score_path),
        "score_file_sha256": sha256(score_path),
        "selection_direction": "larger is selected earlier",
        "elapsed_seconds": time.time() - started,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    write_json(args.output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
