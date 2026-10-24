#!/usr/bin/env python
"""Score additional label/rationale-blind baselines on raw MedGemma.

The source paper specifies architectural MC dropout for BALD/BatchBALD, but
MedGemma-4B-IT has zero architectural dropout.  We therefore use deterministic
MC hidden-state dropout at the input of the final transformer block and record
that adaptation explicitly.  Noise-Stability likewise uses relative hidden-
state perturbations as an executable proxy for the underspecified parameter-
noise baseline in the source paper.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from common import (
    answer_prompt,
    get_transformer_blocks,
    load_model,
    load_tokenizer,
    read_jsonl,
    set_seed,
    write_json,
    write_jsonl,
)
from original_rsp_raw_utils import (
    entropy_rows,
    next_letter_probabilities,
    sanitize_for_acquisition,
    sha256,
)


SENTINEL = -1e30


class HiddenDropoutPreHook:
    """Inverted dropout on the hidden input to one frozen transformer block."""

    def __init__(self, probability: float, seed: int):
        if not 0.0 < probability < 1.0:
            raise ValueError("dropout probability must be in (0, 1)")
        self.probability = float(probability)
        self.seed = int(seed)
        self.generators: dict[str, torch.Generator] = {}

    def _generator(self, tensor: torch.Tensor) -> torch.Generator:
        key = str(tensor.device)
        if key not in self.generators:
            generator = torch.Generator(device=tensor.device)
            generator.manual_seed(self.seed)
            self.generators[key] = generator
        return self.generators[key]

    def __call__(self, _module, inputs):
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return inputs
        hidden = inputs[0]
        keep = torch.rand(
            hidden.shape,
            generator=self._generator(hidden),
            device=hidden.device,
            dtype=torch.float32,
        ) >= self.probability
        perturbed = hidden * keep.to(hidden.dtype) / (1.0 - self.probability)
        return (perturbed,) + inputs[1:]


class RelativeGaussianPreHook:
    """Relative Gaussian noise at the input of one frozen transformer block."""

    def __init__(self, sigma: float, seed: int):
        if sigma <= 0.0:
            raise ValueError("noise sigma must be positive")
        self.sigma = float(sigma)
        self.seed = int(seed)
        self.generators: dict[str, torch.Generator] = {}

    def _generator(self, tensor: torch.Tensor) -> torch.Generator:
        key = str(tensor.device)
        if key not in self.generators:
            generator = torch.Generator(device=tensor.device)
            generator.manual_seed(self.seed)
            self.generators[key] = generator
        return self.generators[key]

    def __call__(self, _module, inputs):
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return inputs
        hidden = inputs[0]
        rms = hidden.float().pow(2).mean(dim=tuple(range(1, hidden.ndim)), keepdim=True)
        rms = rms.sqrt().to(hidden.dtype)
        noise = torch.randn(
            hidden.shape,
            generator=self._generator(hidden),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        return (hidden + self.sigma * rms * noise,) + inputs[1:]


def bald_scores(mc_probabilities: np.ndarray) -> np.ndarray:
    """Mutual information H[E p] - E H[p] for [T,N,C] probabilities."""
    values = np.asarray(mc_probabilities, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"Expected [T,N,C], got {values.shape}")
    return entropy_rows(values.mean(axis=0)) - np.mean(
        np.stack([entropy_rows(item) for item in values], axis=0), axis=0
    )


def _sample_batchbald_log_likelihoods(
    probabilities: np.ndarray,
    samples: int,
    seed: int,
) -> np.ndarray:
    """Return log p_t(y_bi|x_i) with shape [B,N,T]."""
    values = np.asarray(probabilities, dtype=np.float64)
    passes, items, _choices = values.shape
    totals = values.sum(axis=2, keepdims=True)
    values = values / np.clip(totals, 1e-12, None)
    rng = np.random.default_rng(seed)
    sampled_pass = rng.integers(0, passes, size=samples)
    mixture_rows = values[sampled_pass, :, :]
    uniforms = rng.random((samples, items, 1))
    labels = (uniforms > np.cumsum(mixture_rows, axis=2)).sum(axis=2)
    labels = np.minimum(labels, values.shape[2] - 1)
    log_likelihoods = np.empty((samples, items, passes), dtype=np.float32)
    item_indices = np.arange(items)[None, :]
    for pass_index in range(passes):
        selected = values[pass_index][item_indices, labels]
        log_likelihoods[:, :, pass_index] = np.log(
            np.clip(selected, 1e-12, 1.0)
        ).astype(np.float32)
    return log_likelihoods


def batchbald_ranking_score(
    mc_probabilities: np.ndarray,
    select_count: int,
    samples: int,
    seed: int,
    candidate_indices: np.ndarray | None = None,
    device: str | torch.device = "cpu",
    chunk_size: int = 128,
) -> np.ndarray:
    """Greedy BatchBALD with Monte Carlo joint-label entropy estimation.

    Scores are ranks rather than raw MI because the training pipeline only
    needs the ordered selected batch.
    """
    values = np.asarray(mc_probabilities, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"Expected [T,N,C], got {values.shape}")
    total_items = values.shape[1]
    if candidate_indices is None:
        candidate_indices = np.arange(total_items, dtype=np.int64)
    else:
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    subset = values[:, candidate_indices, :]
    count = min(int(select_count), len(candidate_indices))
    if count <= 0:
        raise ValueError("select_count must be positive")

    conditional_entropy = np.mean(
        np.stack([entropy_rows(item) for item in subset], axis=0), axis=0
    ).astype(np.float32)
    log_likelihoods = _sample_batchbald_log_likelihoods(subset, samples, seed)
    log_likelihoods_t = torch.from_numpy(log_likelihoods).to(device)
    conditional_t = torch.from_numpy(conditional_entropy).to(device)
    joint = torch.zeros((samples, subset.shape[0]), dtype=torch.float32, device=device)
    selected = torch.zeros(len(candidate_indices), dtype=torch.bool, device=device)
    selected_conditional = 0.0
    local_order: list[int] = []

    for _rank in range(count):
        best_score = -float("inf")
        best_index = -1
        for start in range(0, len(candidate_indices), chunk_size):
            stop = min(start + chunk_size, len(candidate_indices))
            combined = joint[:, None, :] + log_likelihoods_t[:, start:stop, :]
            mixture_logp = torch.logsumexp(combined, dim=2) - math.log(subset.shape[0])
            joint_entropy = -mixture_logp.mean(dim=0)
            mutual_information = joint_entropy - selected_conditional - conditional_t[start:stop]
            local_selected = selected[start:stop]
            mutual_information = mutual_information.masked_fill(local_selected, -torch.inf)
            chunk_value, chunk_offset = torch.max(mutual_information, dim=0)
            value = float(chunk_value.item())
            index = start + int(chunk_offset.item())
            if value > best_score:
                best_score = value
                best_index = index
        if best_index < 0:
            raise RuntimeError("BatchBALD failed to select a candidate")
        selected[best_index] = True
        local_order.append(best_index)
        joint += log_likelihoods_t[:, best_index, :]
        selected_conditional += float(conditional_entropy[best_index])

    score = np.full(total_items, SENTINEL, dtype=np.float64)
    for rank, local_index in enumerate(local_order):
        score[candidate_indices[local_index]] = float(count - rank)
    return score


def _language_backbone(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    candidates = [
        getattr(getattr(base, "model", None), "language_model", None),
        getattr(base, "model", None),
        getattr(base, "language_model", None),
    ]
    for candidate in candidates:
        if candidate is not None and callable(candidate):
            return candidate
    raise RuntimeError("Could not locate the MedGemma language backbone")


@torch.no_grad()
def pooled_embeddings(
    model,
    tokenizer,
    rows: Sequence[dict],
    batch_size: int,
    max_length: int,
    normalize: bool,
) -> np.ndarray:
    """Mean-pooled last-layer hidden states from acquisition-visible prompts."""
    model.eval()
    device = next(model.parameters()).device
    backbone = _language_backbone(model)
    vectors = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded = tokenizer(
            [answer_prompt(row) for row in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        output = backbone(**encoded, use_cache=False, return_dict=True)
        hidden = output.last_hidden_state.float()
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        if normalize:
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
        vectors.append(pooled.cpu().numpy().astype(np.float32))
        del output, hidden, pooled
    return np.concatenate(vectors, axis=0)


def sift_ranking_score(
    embeddings: np.ndarray,
    uncertainty: np.ndarray,
    select_count: int,
    diversity_weight: float,
) -> np.ndarray:
    """Entropy plus greedy cosine-diversity ranking."""
    count = len(uncertainty)
    selected = np.zeros(count, dtype=bool)
    score = np.full(count, SENTINEL, dtype=np.float64)
    scaled = np.asarray(uncertainty, dtype=np.float64) - float(np.min(uncertainty))
    scaled /= max(1e-12, float(np.max(scaled)))
    min_distance = np.full(count, 2.0, dtype=np.float64)
    for rank in range(min(select_count, count)):
        if rank == 0:
            chosen = int(np.argmax(scaled))
        else:
            combined = scaled + diversity_weight * (min_distance / 2.0)
            combined[selected] = -np.inf
            chosen = int(np.argmax(combined))
        selected[chosen] = True
        score[chosen] = float(select_count - rank)
        distance = 1.0 - np.asarray(embeddings) @ np.asarray(embeddings[chosen])
        min_distance = np.minimum(min_distance, distance)
    return score


def kcenter_ranking_score(
    embeddings: np.ndarray,
    select_count: int,
    seed: int,
) -> np.ndarray:
    """Seeded-first-center greedy cosine k-center ranking."""
    count = len(embeddings)
    if count == 0:
        raise ValueError("No embeddings")
    selected = np.zeros(count, dtype=bool)
    score = np.full(count, SENTINEL, dtype=np.float64)
    first = int(np.random.default_rng(seed).integers(0, count))
    chosen = first
    min_distance = np.full(count, 2.0, dtype=np.float64)
    for rank in range(min(select_count, count)):
        selected[chosen] = True
        score[chosen] = float(select_count - rank)
        distance = 1.0 - np.asarray(embeddings) @ np.asarray(embeddings[chosen])
        min_distance = np.minimum(min_distance, distance)
        min_distance[selected] = -np.inf
        chosen = int(np.argmax(min_distance))
    return score


def resolve_layer(model, offset: int):
    blocks = get_transformer_blocks(model)
    index = offset if offset >= 0 else len(blocks) + offset
    if not 0 <= index < len(blocks):
        raise ValueError(f"Invalid layer offset {offset} for {len(blocks)} blocks")
    return blocks, index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--clean_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="google/medgemma-4b-it")
    parser.add_argument("--cache_dir", default="model_cache")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--embedding_batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--mc_passes", type=int, default=8)
    parser.add_argument("--dropout_probability", type=float, default=0.1)
    parser.add_argument("--dropout_layer", type=int, default=-1)
    parser.add_argument("--batchbald_samples", type=int, default=1000)
    parser.add_argument("--batchbald_preselect", type=int, default=5000)
    parser.add_argument("--noise_passes", type=int, default=8)
    parser.add_argument("--noise_sigma", type=float, default=0.03)
    parser.add_argument("--noise_layer", type=int, default=-1)
    parser.add_argument("--noise_preselect", type=int, default=5000)
    parser.add_argument("--sift_preselect", type=int, default=5000)
    parser.add_argument("--sift_diversity_weight", type=float, default=0.5)
    parser.add_argument(
        "--pool_limit",
        type=int,
        default=0,
        help="Deterministic prefix limit for smoke tests only; zero uses the full pool.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(args.seed)
    started = time.time()
    rows = read_jsonl(args.data_dir / "pool.jsonl")
    if args.pool_limit:
        rows = rows[: args.pool_limit]
    visible = [sanitize_for_acquisition(row) for row in rows]
    if len(rows) < args.budget:
        raise ValueError(f"Pool has {len(rows)} rows, below budget {args.budget}")
    clean_path = args.clean_dir / "clean_scores.npz"
    clean = np.load(clean_path)
    probabilities = np.asarray(clean["probabilities"], dtype=np.float64)[: len(rows)]
    entropy = np.asarray(clean["entropy"], dtype=np.float64)[: len(rows)]
    if probabilities.shape[0] != len(rows):
        raise ValueError("Clean score/pool length mismatch")

    tokenizer = load_tokenizer(args.model_name, args.cache_dir)
    model = load_model(args.model_name, args.cache_dir, None, trainable=False)
    blocks, dropout_index = resolve_layer(model, args.dropout_layer)
    dropout_probabilities = []
    for pass_index in range(args.mc_passes):
        hook = HiddenDropoutPreHook(
            args.dropout_probability,
            args.seed * 1_000_003 + pass_index * 104729,
        )
        handle = blocks[dropout_index].register_forward_pre_hook(hook)
        try:
            dropout_probabilities.append(
                next_letter_probabilities(
                    model, tokenizer, visible, args.batch_size, args.max_length
                )
            )
        finally:
            handle.remove()
        print(f"[extra-baselines] MC dropout {pass_index + 1}/{args.mc_passes}", flush=True)
    dropout_stack = np.stack(dropout_probabilities, axis=0)
    bald = bald_scores(dropout_stack)

    batchbald_count = min(args.batchbald_preselect, len(rows))
    batchbald_candidates = np.argsort(-bald, kind="stable")[:batchbald_count]
    batchbald = batchbald_ranking_score(
        dropout_stack,
        args.budget,
        args.batchbald_samples,
        args.seed + 271828,
        candidate_indices=batchbald_candidates,
        device=next(model.parameters()).device,
    )
    print("[extra-baselines] BatchBALD complete", flush=True)

    sift_count = min(args.sift_preselect, len(rows))
    sift_candidates = np.argsort(-entropy, kind="stable")[:sift_count]
    sift_embeddings = pooled_embeddings(
        model,
        tokenizer,
        [visible[index] for index in sift_candidates],
        args.embedding_batch_size,
        args.max_length,
        normalize=True,
    )
    sift_local = sift_ranking_score(
        sift_embeddings,
        entropy[sift_candidates],
        args.budget,
        args.sift_diversity_weight,
    )
    sift = np.full(len(rows), SENTINEL, dtype=np.float64)
    sift[sift_candidates] = sift_local
    print("[extra-baselines] SIFT complete", flush=True)

    noise_count = min(args.noise_preselect, len(rows))
    noise_candidates = np.random.default_rng(args.seed + 314159).permutation(len(rows))[
        :noise_count
    ]
    noise_rows = [visible[index] for index in noise_candidates]
    clean_embeddings = pooled_embeddings(
        model,
        tokenizer,
        noise_rows,
        args.embedding_batch_size,
        args.max_length,
        normalize=False,
    )
    blocks, noise_index = resolve_layer(model, args.noise_layer)
    squared_delta = np.zeros_like(clean_embeddings, dtype=np.float64)
    for pass_index in range(args.noise_passes):
        hook = RelativeGaussianPreHook(
            args.noise_sigma,
            args.seed * 2_000_003 + pass_index * 130363,
        )
        handle = blocks[noise_index].register_forward_pre_hook(hook)
        try:
            perturbed = pooled_embeddings(
                model,
                tokenizer,
                noise_rows,
                args.embedding_batch_size,
                args.max_length,
                normalize=False,
            )
        finally:
            handle.remove()
        squared_delta += np.square(perturbed.astype(np.float64) - clean_embeddings)
        print(f"[extra-baselines] noise pass {pass_index + 1}/{args.noise_passes}", flush=True)
    sensitivity = np.sqrt(squared_delta / args.noise_passes).astype(np.float32)
    norms = np.linalg.norm(sensitivity, axis=1, keepdims=True)
    sensitivity /= np.clip(norms, 1e-12, None)
    noise_local = kcenter_ranking_score(
        sensitivity,
        args.budget,
        args.seed + 161803,
    )
    noise_stability = np.full(len(rows), SENTINEL, dtype=np.float64)
    noise_stability[noise_candidates] = noise_local

    output_rows = []
    for index, row in enumerate(rows):
        output_rows.append(
            {
                "index": index,
                "uid": row["uid"],
                "bald_hidden_dropout": float(bald[index]),
                "batchbald_hidden_dropout": float(batchbald[index]),
                "noise_stability": float(noise_stability[index]),
                "sift": float(sift[index]),
                "selection_unit_cost": 1,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "scores.jsonl"
    write_jsonl(score_path, output_rows)
    np.savez_compressed(
        args.output_dir / "baseline_scores.npz",
        dropout_probabilities=dropout_stack,
        bald=bald,
        batchbald=batchbald,
        batchbald_candidates=batchbald_candidates,
        sift=sift,
        sift_candidates=sift_candidates,
        noise_stability=noise_stability,
        noise_candidates=noise_candidates,
    )
    metadata = {
        "protocol": "Original RSP--Raw five-dataset direct-baseline extension",
        "dataset": args.dataset,
        "backbone": args.model_name,
        "adapter_loaded": False,
        "scoring_initialization": "raw_backbone_without_answer_only_warmup",
        "selector_visible_fields": ["uid", "question", "options", "domain"],
        "answer_labels_used_during_acquisition": False,
        "rationales_used_during_acquisition": False,
        "budget_items": args.budget,
        "pool_rows": len(rows),
        "methods": {
            "bald_hidden_dropout": {
                "passes": args.mc_passes,
                "dropout_probability": args.dropout_probability,
                "layer": dropout_index,
                "adaptation": (
                    "MC hidden-state dropout replaces architectural MC dropout because "
                    "google/medgemma-4b-it exposes zero architectural dropout"
                ),
            },
            "batchbald_hidden_dropout": {
                "passes": args.mc_passes,
                "joint_label_samples": args.batchbald_samples,
                "preselection_requested": args.batchbald_preselect,
                "preselection_realized": batchbald_count,
                "preselection_score": "BALD",
            },
            "noise_stability": {
                "passes": args.noise_passes,
                "relative_sigma": args.noise_sigma,
                "layer": noise_index,
                "representation": "RMS pooled-hidden-state change across perturbations",
                "selection": "seeded greedy cosine k-center",
                "adaptation": (
                    "relative final-block-input hidden perturbation is an executable proxy "
                    "for the source paper's underspecified parameter perturbation"
                ),
                "preselection_requested": args.noise_preselect,
                "preselection_realized": noise_count,
            },
            "sift": {
                "preselection_requested": args.sift_preselect,
                "preselection_realized": sift_count,
                "preselection_score": "answer entropy",
                "representation": "mean-pooled final hidden state",
                "distance": "cosine",
                "diversity_weight": args.sift_diversity_weight,
            },
        },
        "pool_sha256": sha256(args.data_dir / "pool.jsonl"),
        "clean_score_sha256": sha256(clean_path),
        "score_file_sha256": sha256(score_path),
        "elapsed_seconds": time.time() - started,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    write_json(args.output_dir / "score_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
