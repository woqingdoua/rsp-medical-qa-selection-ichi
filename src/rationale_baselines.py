#!/usr/bin/env python
"""Rationale-visible data-selection baselines and the RSP pilot scorer.

The scorer deliberately separates static likelihood baselines, RSP, and TEMP so
that the expensive components can be run in parallel on Slurm.  Every exported
method uses a "larger is selected earlier" convention.  Raw statistics are kept
alongside the selection score to make directionality auditable.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from common import (
    answer_prompt,
    get_transformer_blocks,
    load_model,
    load_tokenizer,
    option_letters,
    question_block,
    read_jsonl,
    set_seed,
    strip_answer_leakage,
    write_json,
    write_jsonl,
)


PAPER_REFERENCES = {
    "temp": "Jin et al., ICML 2026, arXiv:2606.26797",
    "rsr": "Yang et al., ACL 2026, 2026.acl-long.1950",
    "lalp": "Just et al., arXiv:2510.03988v2 (2026 revision)",
    "complexity_ft": "Goncharov et al., Findings of EACL 2026, 2026.findings-eacl.34",
    "aslec": "Wang et al., Findings of ACL 2026, 2026.findings-acl.918",
}


def split_reasoning_steps(text: str, max_steps: int = 48) -> List[str]:
    """Deterministic sentence/newline segmentation for short medical rationales.

    The original LALP and ASLEC experiments use model-assisted or dataset-native
    steps.  MedMCQA/MedPrompt provide no step annotations, so this deterministic
    adaptation is shared by both baselines and recorded in the metadata.
    """
    text = strip_answer_leakage(text).replace("\r\n", "\n").strip()
    if not text:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n+", text) if part.strip()]
    steps: List[str] = []
    for paragraph in paragraphs:
        pieces = re.split(r"(?<=[.!?])\s+(?=(?:[A-Z0-9\[(]|However\b|Therefore\b|Thus\b))", paragraph)
        steps.extend(piece.strip() for piece in pieces if piece.strip())
    if len(steps) > max_steps:
        head = steps[: max_steps - 1]
        head.append(" ".join(steps[max_steps - 1 :]))
        steps = head
    return steps or [text]


def _reasoning_prefix(row: dict) -> str:
    return question_block(row) + "\nReasoning:\n"


def _encode_steps(
    tokenizer,
    row: dict,
    max_length: int,
    max_rationale_tokens: int,
) -> Tuple[List[int], List[List[int]]]:
    prompt_ids = tokenizer.encode(_reasoning_prefix(row), add_special_tokens=False)
    if len(prompt_ids) >= max_length - 2:
        keep = max_length - 2
        head = keep // 2
        prompt_ids = prompt_ids[:head] + prompt_ids[-(keep - head) :]
    available = min(max_rationale_tokens, max_length - len(prompt_ids))
    encoded_steps: List[List[int]] = []
    remaining = available
    for index, step in enumerate(split_reasoning_steps(row["rationale"])):
        prefix = "" if index == 0 else "\n"
        ids = tokenizer.encode(prefix + step, add_special_tokens=False)
        if not ids:
            continue
        ids = ids[:remaining]
        if ids:
            encoded_steps.append(ids)
            remaining -= len(ids)
        if remaining <= 0:
            break
    return prompt_ids, encoded_steps


@torch.no_grad()
def _response_token_stats(model, prompt_ids: Sequence[int], step_ids: Sequence[Sequence[int]]) -> dict:
    device = next(model.parameters()).device
    response_ids = [token for step in step_ids for token in step]
    if not response_ids:
        return {
            "logp": np.empty(0),
            "rank": np.empty(0),
            "step": np.empty(0, dtype=np.int64),
            "first": np.empty(0, dtype=bool),
        }
    ids = list(prompt_ids) + response_ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    start = len(prompt_ids)
    logits = output.logits[0, start - 1 : len(ids) - 1].float()
    target = torch.tensor(response_ids, dtype=torch.long, device=device)
    target_logits = logits.gather(1, target[:, None]).squeeze(1)
    logp = target_logits - torch.logsumexp(logits, dim=-1)
    rank = 1 + (logits > target_logits[:, None]).sum(dim=-1)
    step_index: List[int] = []
    first: List[bool] = []
    for index, step in enumerate(step_ids):
        step_index.extend([index] * len(step))
        first.extend([True] + [False] * (len(step) - 1))
    return {
        "logp": logp.cpu().numpy().astype(np.float64),
        "rank": rank.cpu().numpy().astype(np.float64),
        "step": np.asarray(step_index, dtype=np.int64),
        "first": np.asarray(first, dtype=bool),
    }


@torch.no_grad()
def _answer_vocab_entropy(model, tokenizer, row: dict, max_length: int) -> float:
    device = next(model.parameters()).device
    ids = tokenizer.encode(answer_prompt(row), add_special_tokens=False)
    if len(ids) > max_length:
        head = max_length // 2
        ids = ids[:head] + ids[-(max_length - head) :]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False).logits[
        0, -1
    ].float()
    logp = F.log_softmax(logits, dim=-1)
    probs = logp.exp()
    return float(-(probs * logp).sum().item())


@torch.no_grad()
def _answer_choice_uncertainty(
    model,
    tokenizer,
    row: dict,
    max_length: int,
    letter_ids: Sequence[int],
) -> Tuple[float, float]:
    """Candidate entropy and negative top-two margin; larger means more uncertain."""
    device = next(model.parameters()).device
    ids = tokenizer.encode(answer_prompt(row), add_special_tokens=False)
    if len(ids) > max_length:
        head = max_length // 2
        ids = ids[:head] + ids[-(max_length - head) :]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False).logits[
        0, -1
    ].float()
    values = logits[torch.tensor(letter_ids, dtype=torch.long, device=device)]
    probs = values.softmax(dim=0)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    top_two = torch.topk(probs, k=2).values
    return float(entropy.item()), float(-(top_two[0] - top_two[1]).item())


@torch.no_grad()
def _local_average_logp(
    model,
    prompt_ids: Sequence[int],
    steps: Sequence[Sequence[int]],
    context_fraction: float,
    max_length: int,
) -> float:
    device = next(model.parameters()).device
    step_scores: List[float] = []
    for index, current in enumerate(steps):
        if not current:
            continue
        if index == 0:
            context: List[int] = []
        else:
            k = max(1, int(math.ceil(context_fraction * index)))
            context = [token for step in steps[max(0, index - k) : index] for token in step]
        available_context = max(0, max_length - len(prompt_ids) - len(current))
        context = context[-available_context:]
        ids = list(prompt_ids) + context + list(current)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False).logits
        start = len(prompt_ids) + len(context)
        token_logits = logits[0, start - 1 : len(ids) - 1].float()
        target = torch.tensor(current, dtype=torch.long, device=device)
        logp = token_logits.gather(1, target[:, None]).squeeze(1) - torch.logsumexp(
            token_logits, dim=-1
        )
        step_scores.append(float(logp.mean().item()))
    return float(np.mean(step_scores)) if step_scores else float("-inf")


def _finite_mean(values: np.ndarray, fallback: float) -> float:
    return float(values.mean()) if len(values) else fallback


def score_static(model, tokenizer, rows: Sequence[dict], args) -> Tuple[List[dict], dict]:
    model.eval()
    letter_id_cache: Dict[Tuple[str, ...], List[int]] = {}
    output: List[dict] = []
    for index, row in enumerate(rows):
        letters = tuple(option_letters(row))
        if letters not in letter_id_cache:
            letter_id_cache[letters] = _letter_token_ids(tokenizer, letters)
        letter_ids = letter_id_cache[letters]
        prompt_ids, steps = _encode_steps(
            tokenizer, row, args.max_length, args.max_rationale_tokens
        )
        stats = _response_token_stats(model, prompt_ids, steps)
        logp = stats["logp"]
        first = stats["first"]
        if len(logp) == 0:
            payload = {"uid": row["uid"]}
            for key in (
                "complexity_ft",
                "rsr",
                "lalp",
                "aslec_drop",
                "aslec_casl",
                "global_logp",
                "answer_entropy",
                "answer_margin",
            ):
                payload[key] = float("-inf")
            output.append(payload)
            continue
        global_logp = float(logp.mean())
        first_logp = _finite_mean(logp[first], global_logp)
        drop_logp = _finite_mean(logp[~first], global_logp)
        z = float(first.sum() / len(first))
        raw_rsr = float(np.minimum(stats["rank"], args.rsr_rank_clip).sum() / np.maximum(-logp.sum(), 1e-8))
        lalp = _local_average_logp(
            model, prompt_ids, steps, args.lalp_context_fraction, args.max_length
        )
        entropy = _answer_vocab_entropy(model, tokenizer, row, args.max_length)
        answer_entropy, answer_margin = _answer_choice_uncertainty(
            model, tokenizer, row, args.max_length, letter_ids
        )
        output.append(
            {
                "uid": row["uid"],
                "random": float(np.random.default_rng(args.seed + index * 104729).random()),
                "complexity_ft": entropy,
                "answer_entropy": answer_entropy,
                "answer_margin": answer_margin,
                "rsr": -raw_rsr,
                "rsr_raw": raw_rsr,
                "lalp": lalp,
                "aslec_drop": drop_logp,
                "global_logp": global_logp,
                "aslec_first_logp": first_logp,
                "aslec_first_ratio": z,
                "rationale_tokens_scored": int(len(logp)),
                "rationale_steps": int(first.sum()),
            }
        )
        if (index + 1) % args.log_every == 0:
            print(f"[static] {index + 1}/{len(rows)}", flush=True)

    valid = [item for item in output if np.isfinite(item.get("global_logp", -np.inf))]
    if valid:
        x = np.asarray(
            [
                [item["aslec_first_logp"], item["aslec_drop"], item["aslec_first_ratio"]]
                for item in valid
            ],
            dtype=np.float64,
        )
        y = np.asarray([item["global_logp"] for item in valid], dtype=np.float64)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        gamma = float(beta[2])
        for item in valid:
            item["aslec_casl"] = float(item["global_logp"] - gamma * item["aslec_first_ratio"])
    else:
        beta = np.full(3, np.nan)
        gamma = float("nan")
    metadata = {
        "component": "static",
        "methods": [
            "random",
            "answer_entropy",
            "answer_margin",
            "complexity_ft",
            "rsr",
            "lalp",
            "aslec_drop",
            "aslec_casl",
        ],
        "aslec_ols_beta": beta.tolist(),
        "aslec_gamma": gamma,
        "step_segmentation": "deterministic newline + sentence heuristic shared by LALP and ASLEC",
        "lalp_context_fraction": args.lalp_context_fraction,
        "rsr_rank_clip": args.rsr_rank_clip,
        "selection_direction": "larger is better; rsr is stored as negative raw RSR",
    }
    return output, metadata


def _masked_hidden_noise_hook(mask: torch.Tensor, alpha: float, seed: int):
    def hook(_module, _inputs, output):
        def perturb(tensor: torch.Tensor) -> torch.Tensor:
            local_mask = mask.to(device=tensor.device, dtype=tensor.dtype)
            if local_mask.shape[1] != tensor.shape[1]:
                raise RuntimeError(
                    f"RSP mask length {local_mask.shape[1]} != hidden length {tensor.shape[1]}"
                )
            denom = local_mask.float().sum() * tensor.shape[-1]
            rms = ((tensor.float() * local_mask.float()).pow(2).sum() / denom.clamp_min(1.0)).sqrt()
            generator = torch.Generator(device=tensor.device)
            generator.manual_seed(seed)
            noise = torch.randn(
                tensor.shape, generator=generator, dtype=tensor.dtype, device=tensor.device
            )
            return tensor + alpha * rms.to(tensor.dtype) * noise * local_mask

        if isinstance(output, torch.Tensor):
            return perturb(output)
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return (perturb(output[0]),) + output[1:]
        if isinstance(output, list) and output and isinstance(output[0], torch.Tensor):
            return [perturb(output[0])] + output[1:]
        return output

    return hook


def _letter_token_ids(tokenizer, letters: Sequence[str]) -> List[int]:
    encoded = [tokenizer.encode(" " + letter, add_special_tokens=False) for letter in letters]
    if not all(len(ids) == 1 for ids in encoded):
        raise RuntimeError(f"RSP expects single-token option letters after a space; observed {encoded}")
    return [ids[0] for ids in encoded]


def _resolve_rsp_layers(block_count: int, args) -> Tuple[List[str], List[int]]:
    """Resolve a prespecified depth profile against the model block count.

    Explicit/deep preserve the canonical offsets.  Fractional profiles are
    defined before looking at outcomes so the layer ablation cannot tune on
    downstream test performance.
    """
    profile = args.rsp_layer_profile
    if profile == "explicit":
        labels = [str(value) for value in args.rsp_layers]
        indices = [
            value if value >= 0 else block_count + value for value in args.rsp_layers
        ]
    elif profile == "deep":
        offsets = (-2, -4, -8)
        labels = [str(value) for value in offsets]
        indices = [block_count + value for value in offsets]
    else:
        fractions = {
            "shallow": (0.20, 0.30, 0.40),
            "middle": (0.45, 0.55, 0.65),
            "mixed": (0.30, 0.60, None),
        }[profile]
        indices = [
            block_count - 2 if fraction is None else int(round(fraction * (block_count - 1)))
            for fraction in fractions
        ]
        labels = [
            "L-2" if fraction is None else f"{fraction:.2f}L"
            for fraction in fractions
        ]
    if any(index < 0 or index >= block_count for index in indices):
        raise ValueError(
            f"Invalid RSP profile {profile}: resolved {indices} for {block_count} blocks"
        )
    if len(set(indices)) != len(indices):
        raise ValueError(
            f"RSP profile {profile} contains duplicate resolved layers {indices} "
            f"for {block_count} blocks"
        )
    return labels, indices


@torch.no_grad()
def _rationale_margin(
    model,
    token_ids: Sequence[int],
    letter_ids: Sequence[int],
    gold_index: int,
) -> float:
    device = next(model.parameters()).device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False).logits[
        0, -1
    ].float()
    values = logits[torch.tensor(letter_ids, dtype=torch.long, device=device)]
    distractor = torch.cat((values[:gold_index], values[gold_index + 1 :])).max()
    return float((values[gold_index] - distractor).item())


def score_rsp(model, tokenizer, rows: Sequence[dict], args) -> Tuple[List[dict], dict]:
    model.eval()
    blocks = get_transformer_blocks(model)
    layer_labels, layer_indices = _resolve_rsp_layers(len(blocks), args)
    letter_id_cache: Dict[Tuple[str, ...], List[int]] = {}
    suffix_ids = tokenizer.encode("\nFinal Answer:", add_special_tokens=False)
    output: List[dict] = []
    for row_index, row in enumerate(rows):
        letters = tuple(option_letters(row))
        if letters not in letter_id_cache:
            letter_id_cache[letters] = _letter_token_ids(tokenizer, letters)
        letter_ids = letter_id_cache[letters]
        prompt_ids, steps = _encode_steps(
            tokenizer,
            row,
            args.max_length - len(suffix_ids),
            args.max_rationale_tokens,
        )
        rationale_ids = [token for step in steps for token in step]
        token_ids = prompt_ids + rationale_ids + suffix_ids
        mask = torch.zeros((1, len(token_ids), 1), dtype=torch.float32)
        mask[:, len(prompt_ids) : len(prompt_ids) + len(rationale_ids), :] = 1.0
        gold_index = letters.index(row["answer_letter"])
        clean_margin = _rationale_margin(model, token_ids, letter_ids, gold_index)
        drops: List[float] = []
        layer_means: Dict[str, float] = {}
        layer_raw: Dict[str, List[float]] = {}
        for layer_label, layer_index in zip(layer_labels, layer_indices):
            local_drops: List[float] = []
            for pass_index in range(args.rsp_passes):
                noise_seed = args.seed * 1_000_003 + row_index * 1009 + layer_index * 37 + pass_index
                handle = blocks[layer_index].register_forward_hook(
                    _masked_hidden_noise_hook(mask, args.rsp_alpha, noise_seed)
                )
                try:
                    noisy_margin = _rationale_margin(model, token_ids, letter_ids, gold_index)
                finally:
                    handle.remove()
                local_drops.append(clean_margin - noisy_margin)
            drops.extend(local_drops)
            layer_key = str(layer_index)
            layer_means[layer_key] = float(np.mean(local_drops))
            layer_raw[layer_key] = [float(value) for value in local_drops]
        drop_array = np.asarray(drops, dtype=np.float64)
        positive_fraction = float(np.mean(drop_array > 0.0))
        negative_fraction = float(np.mean(drop_array < 0.0))
        consistency = max(positive_fraction, negative_fraction)
        positive_mean = float(np.maximum(drop_array, 0.0).mean())
        # The application asks how strongly rationale-local perturbations affect
        # the decision boundary, not only whether one arbitrary Gaussian draw
        # lowers it. RMS retains that magnitude while signed diagnostics expose
        # whether perturbations tend to help or hurt the gold option.
        rsp = float(np.sqrt(np.mean(np.square(drop_array))))
        payload = {
            "uid": row["uid"],
            "rsp": rsp,
            "rsp_clean_margin": clean_margin,
            "rsp_mean_signed_drop": float(drop_array.mean()),
            "rsp_positive_drop_mean": positive_mean,
            "rsp_direction_consistency": consistency,
            "rsp_positive_fraction": positive_fraction,
            "rsp_negative_fraction": negative_fraction,
            "rsp_worst_drop": float(drop_array.max()),
            "rsp_layer_mean_drops": layer_means,
        }
        if args.rsp_store_raw_drops:
            payload["rsp_raw_drops"] = [float(value) for value in drops]
            payload["rsp_layer_raw_drops"] = layer_raw
        output.append(payload)
        if (row_index + 1) % args.log_every == 0:
            print(f"[rsp] {row_index + 1}/{len(rows)}", flush=True)
    metadata = {
        "component": "rsp",
        "method": "rsp",
        "definition": "RMS shift of the gold-vs-best-distractor log-probability margin",
        "noise_scope": "rationale tokens only",
        "relative_noise_alpha": args.rsp_alpha,
        "layer_profile": args.rsp_layer_profile,
        "layer_labels": layer_labels,
        "resolved_layer_indices": layer_indices,
        "transformer_block_count": len(blocks),
        "layers": args.rsp_layers if args.rsp_layer_profile == "explicit" else layer_labels,
        "passes_per_layer": args.rsp_passes,
        "raw_drops_stored": args.rsp_store_raw_drops,
    }
    return output, metadata


def _canonical_adapter_key(key: str) -> str:
    key = key.replace(".default.", ".")
    key = re.sub(r"\.(?:temp_direction|adapter)\.", ".", key)
    return key


def _load_adapter_target_state(path: str) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    adapter_path = Path(path)
    weights = adapter_path / "adapter_model.safetensors" if adapter_path.is_dir() else adapter_path
    if not weights.exists():
        raise FileNotFoundError(f"TEMP direction weights not found: {weights}")
    return {_canonical_adapter_key(key): value for key, value in load_file(str(weights)).items()}


def _adapter_parameter_triplets(model, direction_path: str):
    target = _load_adapter_target_state(direction_path)
    triplets = []
    missing = []
    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            continue
        key = _canonical_adapter_key(name)
        if key not in target:
            missing.append(key)
            continue
        base = parameter.detach().clone()
        direction = target[key].to(device=parameter.device, dtype=parameter.dtype) - base
        triplets.append((parameter, base, direction))
    if missing or not triplets:
        raise RuntimeError(
            f"Could not align TEMP adapters: matched={len(triplets)}, missing={len(missing)}, "
            f"first_missing={missing[:3]}"
        )
    return triplets


def _set_adapter_state(triplets, scale: float = 0.0, direction_noise_seed: int | None = None) -> None:
    with torch.no_grad():
        for index, (parameter, base, direction) in enumerate(triplets):
            if direction_noise_seed is None:
                parameter.copy_(base + scale * direction)
            else:
                generator = torch.Generator(device=parameter.device)
                generator.manual_seed(direction_noise_seed + index * 1009)
                noise = torch.randn(
                    direction.shape,
                    generator=generator,
                    device=direction.device,
                    dtype=direction.dtype,
                )
                parameter.copy_(base + scale * (1.0 + noise) * direction)


def _backbone_noise_targets(model):
    targets = []
    for name, parameter in model.named_parameters():
        if "lora_" in name or not parameter.is_floating_point() or parameter.ndim < 2:
            continue
        rms = parameter.detach().float().pow(2).mean().sqrt().clamp_min(1e-8).to(parameter.dtype)
        targets.append((parameter, rms))
    if not targets:
        raise RuntimeError("No backbone parameters found for TEMP random perturbation")
    return targets


def _apply_relative_weight_noise(targets, sigma: float, seed: int, sign: float) -> None:
    """Add/remove the same per-tensor RMS-scaled Gaussian backbone perturbation."""
    with torch.no_grad():
        for index, (parameter, rms) in enumerate(targets):
            generator = torch.Generator(device=parameter.device)
            generator.manual_seed(seed + index * 1009)
            noise = torch.randn(
                parameter.shape,
                generator=generator,
                device=parameter.device,
                dtype=parameter.dtype,
            )
            parameter.add_(sign * sigma * rms * noise)


@torch.no_grad()
def _summed_prefix_losses(
    model,
    tokenizer,
    rows: Sequence[dict],
    max_length: int,
    token_limit: int,
    log_every: int = 0,
    label: str = "loss",
) -> np.ndarray:
    losses = np.full(len(rows), np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        prompt_ids, steps = _encode_steps(tokenizer, row, max_length, token_limit)
        stats = _response_token_stats(model, prompt_ids, steps)
        if len(stats["logp"]):
            losses[index] = float(-stats["logp"].sum())
        if log_every and (index + 1) % log_every == 0:
            print(f"[{label}] {index + 1}/{len(rows)}", flush=True)
    return losses


def _calibrate_random_noise(
    model, tokenizer, rows, noise_targets, args, clean: np.ndarray
) -> Tuple[float, float]:
    sigma = args.temp_random_sigma
    ratio = 1.0
    for attempt in range(args.temp_calibration_steps):
        _apply_relative_weight_noise(noise_targets, sigma, args.seed + 7001, +1.0)
        try:
            noisy = _summed_prefix_losses(
                model, tokenizer, rows, args.max_length, args.temp_difficulty_tokens
            )
        finally:
            _apply_relative_weight_noise(noise_targets, sigma, args.seed + 7001, -1.0)
        ratio = float(np.nanmean(noisy) / np.maximum(np.nanmean(clean), 1e-8))
        print(f"[temp] random sigma={sigma:.6g}, calibration loss ratio={ratio:.3f}", flush=True)
        if args.temp_random_target_low <= ratio <= args.temp_random_target_high:
            break
        if attempt + 1 < args.temp_calibration_steps:
            sigma *= 1.8 if ratio < args.temp_random_target_low else 0.65
    return sigma, ratio


def _calibrate_direction_scale(model, tokenizer, rows, triplets, args, clean: np.ndarray) -> Tuple[float, float]:
    scale = args.temp_direction_scale
    ratio = 1.0
    for attempt in range(args.temp_calibration_steps):
        _set_adapter_state(triplets, scale, args.seed + 9001)
        perturbed = _summed_prefix_losses(
            model, tokenizer, rows, args.max_length, args.temp_reasoning_tokens
        )
        ratio = float(np.nanmean(perturbed) / np.maximum(np.nanmean(clean), 1e-8))
        print(f"[temp] direction scale={scale:.6g}, calibration loss ratio={ratio:.3f}", flush=True)
        if 2.5 <= ratio <= 3.5:
            break
        if attempt + 1 < args.temp_calibration_steps:
            scale *= 1.8 if ratio < 2.5 else 0.65
    _set_adapter_state(triplets)
    return scale, ratio


def _two_cluster_high(values: np.ndarray, seed: int) -> np.ndarray:
    if len(values) < 2 or np.allclose(values, values[0]):
        return np.ones(len(values), dtype=bool)
    labels = KMeans(n_clusters=2, random_state=seed, n_init=10).fit_predict(values[:, None])
    means = [float(values[labels == label].mean()) for label in (0, 1)]
    return labels == int(np.argmax(means))


def _allocate_budget(capacities: Sequence[int], weights: Sequence[float], total: int) -> List[int]:
    capacities = np.asarray(capacities, dtype=np.int64)
    if capacities.sum() < total:
        raise ValueError(f"TEMP difficult pool has {capacities.sum()} examples for budget {total}")
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 1e-12)
    allocation = np.zeros(len(capacities), dtype=np.int64)
    for _ in range(total):
        available = allocation < capacities
        priorities = np.where(available, weights / (allocation + 1.0), -np.inf)
        allocation[int(np.argmax(priorities))] += 1
    return allocation.tolist()


def _temp_select(
    rows: Sequence[dict],
    clean100: np.ndarray,
    noisy100: np.ndarray,
    directional: np.ndarray,
    budget: int,
    seed: int,
    per_cluster: int,
) -> Tuple[List[int], dict]:
    source_to_indices: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        source = str(row.get("subject") or row.get("topic") or "unknown")
        source_to_indices[source].append(index)
    difficult: Dict[str, List[int]] = {}
    source_weights: Dict[str, float] = {}
    for source, indices in source_to_indices.items():
        values = noisy100[indices]
        keep = _two_cluster_high(values, seed + len(difficult) * 17)
        chosen = [index for index, flag in zip(indices, keep) if flag]
        if not chosen:
            chosen = indices
        difficult[source] = chosen
        d_in = float(np.nanmean(noisy100[chosen]))
        d_br = float(np.nanmean(noisy100[chosen] - clean100[chosen]))
        # Eq. 4 uses a softmax over this geometric-mean difficulty.  Store
        # logits here and exponentiate after subtracting the maximum to avoid
        # overflow for summed token losses.
        source_weights[source] = float(math.sqrt(max(d_in, 0.0) * max(d_br, 0.0)))
    sources = sorted(difficult)
    source_logits = np.asarray([source_weights[source] for source in sources], dtype=np.float64)
    stable_weights = np.exp(source_logits - source_logits.max())
    # Small datasets can contain fewer high-difficulty examples than TEMP's
    # requested internal example budget. Rank every available difficult item
    # in that case; downstream selection still applies the shared token budget.
    effective_budget = min(budget, sum(len(difficult[source]) for source in sources))
    allocations = _allocate_budget(
        [len(difficult[source]) for source in sources],
        stable_weights,
        effective_budget,
    )
    selected: List[int] = []
    source_details = {}
    for source_index, (source, source_budget) in enumerate(zip(sources, allocations)):
        candidates = difficult[source]
        if source_budget == 0:
            continue
        vectors = directional[:, candidates].T
        clusters = max(1, min(len(candidates), source_budget // per_cluster or 1))
        if clusters == 1:
            labels = np.zeros(len(candidates), dtype=np.int64)
        else:
            labels = KMeans(
                n_clusters=clusters, random_state=seed + source_index * 101, n_init=10
            ).fit_predict(vectors)
        cluster_groups = [[candidates[i] for i in np.flatnonzero(labels == label)] for label in range(clusters)]
        cluster_allocations = _allocate_budget(
            [len(group) for group in cluster_groups], [1.0] * clusters, source_budget
        )
        source_selected: List[int] = []
        brittle = directional[-1] - directional[0]
        for group, count in zip(cluster_groups, cluster_allocations):
            ranked = sorted(group, key=lambda index: (-brittle[index], index))
            source_selected.extend(ranked[:count])
        selected.extend(source_selected)
        source_details[source] = {
            "available": len(source_to_indices[source]),
            "difficult": len(candidates),
            "allocated": source_budget,
            "clusters": clusters,
            "difficulty_logit": source_weights[source],
            "softmax_weight": float(stable_weights[sources.index(source)]),
        }
    selected.sort(key=lambda index: (-(directional[-1, index] - directional[0, index]), index))
    return selected, source_details


def score_temp(model, tokenizer, rows: Sequence[dict], args) -> Tuple[List[dict], dict]:
    model.eval()
    triplets = _adapter_parameter_triplets(model, args.temp_direction_path)
    noise_targets = _backbone_noise_targets(model)
    calibration_rows = rows[: min(args.temp_calibration_samples, len(rows))]
    clean100_cal = _summed_prefix_losses(
        model,
        tokenizer,
        calibration_rows,
        args.max_length,
        args.temp_difficulty_tokens,
    )
    sigma, random_ratio = _calibrate_random_noise(
        model, tokenizer, calibration_rows, noise_targets, args, clean100_cal
    )
    clean100 = _summed_prefix_losses(
        model,
        tokenizer,
        rows,
        args.max_length,
        args.temp_difficulty_tokens,
        args.log_every,
        "temp-clean100",
    )
    _apply_relative_weight_noise(noise_targets, sigma, args.seed + 7001, +1.0)
    try:
        noisy100 = _summed_prefix_losses(
            model,
            tokenizer,
            rows,
            args.max_length,
            args.temp_difficulty_tokens,
            args.log_every,
            "temp-random100",
        )
    finally:
        _apply_relative_weight_noise(noise_targets, sigma, args.seed + 7001, -1.0)

    clean_reasoning_cal = _summed_prefix_losses(
        model,
        tokenizer,
        calibration_rows,
        args.max_length,
        args.temp_reasoning_tokens,
    )
    max_scale, direction_ratio = _calibrate_direction_scale(
        model, tokenizer, calibration_rows, triplets, args, clean_reasoning_cal
    )
    first_scale = max_scale / max(args.temp_checkpoints, 1) / 3.0
    scales = np.linspace(first_scale, max_scale, args.temp_checkpoints)
    directional = np.full((args.temp_checkpoints, len(rows)), np.nan, dtype=np.float64)
    for checkpoint, scale in enumerate(scales):
        _set_adapter_state(triplets, float(scale), args.seed + 9001 + checkpoint * 100003)
        directional[checkpoint] = _summed_prefix_losses(
            model,
            tokenizer,
            rows,
            args.max_length,
            args.temp_reasoning_tokens,
            args.log_every,
            f"temp-direction-{checkpoint + 1}",
        )
    _set_adapter_state(triplets)
    selected, source_details = _temp_select(
        rows,
        clean100,
        noisy100,
        directional,
        args.temp_budget,
        args.seed,
        args.temp_per_cluster,
    )
    rank_lookup = {index: rank for rank, index in enumerate(selected)}
    output = []
    for index, row in enumerate(rows):
        output.append(
            {
                "uid": row["uid"],
                "temp": float(args.temp_budget - rank_lookup[index]) if index in rank_lookup else float("-inf"),
                "temp_selected": index in rank_lookup,
                "temp_clean_loss_100": float(clean100[index]),
                "temp_random_loss_100": float(noisy100[index]),
                "temp_brittle_loss": float(directional[-1, index] - directional[0, index]),
                "temp_directional_losses": directional[:, index].tolist(),
            }
        )
    metadata = {
        "component": "temp",
        "method": "TEMP adapted to the shared LoRA/MedGemma protocol",
        "parameter_space": (
            "RMS-scaled Gaussian perturbation of frozen backbone matrices for difficulty; "
            "LoRA rationale-SFT direction for extrapolated checkpoints"
        ),
        "selected_uids": [rows[index]["uid"] for index in selected],
        "budget": args.temp_budget,
        "effective_budget": len(selected),
        "difficulty_tokens": args.temp_difficulty_tokens,
        "reasoning_tokens": args.temp_reasoning_tokens,
        "checkpoints": args.temp_checkpoints,
        "checkpoint_scales": scales.tolist(),
        "random_sigma": sigma,
        "random_calibration_ratio": random_ratio,
        "direction_max_scale": max_scale,
        "direction_calibration_ratio": direction_ratio,
        "source_details": source_details,
    }
    return output, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=("static", "rsp", "temp"), required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_name", default="google/medgemma-4b-it")
    parser.add_argument("--cache_dir", default="model_cache")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--pool_limit", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_rationale_tokens", type=int, default=768)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--rsr_rank_clip", type=int, default=100)
    parser.add_argument("--lalp_context_fraction", type=float, default=0.25)
    parser.add_argument("--rsp_layers", type=int, nargs="+", default=[-2, -4, -8])
    parser.add_argument(
        "--rsp_layer_profile",
        choices=("explicit", "deep", "shallow", "middle", "mixed"),
        default="explicit",
        help=(
            "Prespecified layer-depth profile. explicit uses --rsp_layers; deep is "
            "the canonical -2/-4/-8 profile."
        ),
    )
    parser.add_argument("--rsp_passes", type=int, default=2)
    parser.add_argument("--rsp_alpha", type=float, default=0.10)
    parser.add_argument("--rsp_store_raw_drops", action="store_true")
    parser.add_argument("--temp_direction_path")
    parser.add_argument("--temp_budget", type=int, default=32)
    parser.add_argument("--temp_difficulty_tokens", type=int, default=100)
    parser.add_argument("--temp_reasoning_tokens", type=int, default=768)
    parser.add_argument("--temp_checkpoints", type=int, default=4)
    parser.add_argument("--temp_per_cluster", type=int, default=4)
    parser.add_argument("--temp_calibration_samples", type=int, default=8)
    parser.add_argument("--temp_calibration_steps", type=int, default=6)
    parser.add_argument("--temp_random_sigma", type=float, default=0.02)
    parser.add_argument("--temp_random_target_low", type=float, default=2.0)
    parser.add_argument("--temp_random_target_high", type=float, default=3.0)
    parser.add_argument("--temp_direction_scale", type=float, default=1.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for rationale scoring")
    if args.component == "temp" and not args.temp_direction_path:
        parser.error("--temp_direction_path is required for TEMP")
    set_seed(args.seed)
    started = time.time()
    rows = read_jsonl(Path(args.data_dir) / "pool.jsonl")
    if args.pool_limit:
        rows = rows[: args.pool_limit]
    tokenizer = load_tokenizer(args.model_name, args.cache_dir)
    model = load_model(args.model_name, args.cache_dir, args.adapter_path, trainable=False)
    if args.component == "static":
        scores, metadata = score_static(model, tokenizer, rows, args)
    elif args.component == "rsp":
        scores, metadata = score_rsp(model, tokenizer, rows, args)
    else:
        scores, metadata = score_temp(model, tokenizer, rows, args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / f"{args.component}_scores.jsonl", scores)
    metadata.update(
        {
            "dataset": args.dataset,
            "rows": len(rows),
            "model_name": args.model_name,
            "adapter_path": args.adapter_path,
            "rationales_visible_during_scoring": True,
            "references": PAPER_REFERENCES,
            "elapsed_seconds": time.time() - started,
            "arguments": vars(args),
        }
    )
    write_json(output / f"{args.component}_metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
