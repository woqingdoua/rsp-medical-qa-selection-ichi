#!/usr/bin/env python
"""Utilities for the paper-faithful RSP score on an unadapted backbone."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from common import option_letters, question_block, strip_answer_leakage


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_for_acquisition(row: dict) -> dict:
    """Allow-list fields available to the label/rationale-blind selector."""
    return {
        "uid": row["uid"],
        "question": row["question"],
        "options": dict(row["options"]),
        "domain": row.get("domain", "medical"),
    }


def rationale_context_prompt(row: dict) -> str:
    """Put acquired CoT in the prompt; the target remains the final answer only."""
    rationale = strip_answer_leakage(str(row["rationale"]))
    if not rationale:
        raise ValueError(f"Empty rationale for {row.get('uid')}")
    return (
        question_block(row)
        + "\nReasoning:\n"
        + rationale
        + "\nUsing the reasoning above, return only the final option letter.\nFinal Answer:"
    )


def _letter_token_ids(tokenizer, letters: Sequence[str]) -> list[int]:
    encoded = [tokenizer.encode(" " + letter, add_special_tokens=False) for letter in letters]
    if not all(len(value) == 1 for value in encoded):
        raise RuntimeError(f"Option letters must be single tokens; observed {encoded}")
    return [value[0] for value in encoded]


def _truncate_prompt(ids: list[int], max_length: int) -> list[int]:
    if len(ids) <= max_length:
        return ids
    head = max_length // 2
    return ids[:head] + ids[-(max_length - head) :]


class AbsoluteGaussianHook:
    """Independent absolute Gaussian draws from one persistent RNG stream."""

    def __init__(self, sigma: float, seed: int):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
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

    def __call__(self, _module, _inputs, output):
        def perturb(tensor: torch.Tensor) -> torch.Tensor:
            noise = torch.randn(
                tensor.shape,
                generator=self._generator(tensor),
                device=tensor.device,
                dtype=tensor.dtype,
            )
            return tensor + self.sigma * noise

        if isinstance(output, torch.Tensor):
            return perturb(output)
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return (perturb(output[0]),) + output[1:]
        if isinstance(output, list) and output and isinstance(output[0], torch.Tensor):
            return [perturb(output[0])] + output[1:]
        return output


@torch.no_grad()
def next_letter_probabilities(
    model,
    tokenizer,
    rows: Sequence[dict],
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Candidate conditional probabilities from a single coherent forward pass."""
    model.eval()
    if not rows:
        return np.empty((0, 0), dtype=np.float64)
    row_letters = [tuple(option_letters(row)) for row in rows]
    width = max(len(letters) for letters in row_letters)
    probabilities = np.zeros((len(rows), width), dtype=np.float64)
    groups: dict[tuple[str, ...], list[int]] = {}
    for index, letters in enumerate(row_letters):
        groups.setdefault(letters, []).append(index)

    device = next(model.parameters()).device
    for letters, indices in groups.items():
        option_ids = torch.tensor(
            _letter_token_ids(tokenizer, letters), dtype=torch.long, device=device
        )
        for start in range(0, len(indices), batch_size):
            local_indices = indices[start : start + batch_size]
            encoded = [
                _truncate_prompt(
                    tokenizer.encode(
                        # Keep this local to guarantee scoring never sees the answer/rationale.
                        question_block(rows[index])
                        + "\nChoose one option. Return only the final option letter.\nFinal Answer:",
                        add_special_tokens=False,
                    ),
                    max_length,
                )
                for index in local_indices
            ]
            longest = max(len(value) for value in encoded)
            padded = [
                [tokenizer.pad_token_id] * (longest - len(value)) + value for value in encoded
            ]
            masks = [[0] * (longest - len(value)) + [1] * len(value) for value in encoded]
            input_ids = torch.tensor(padded, dtype=torch.long, device=device)
            attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[:, -1, :].float()
            selected = logits.index_select(-1, option_ids)
            local = torch.softmax(selected, dim=-1).cpu().numpy().astype(np.float64)
            probabilities[np.asarray(local_indices), : len(letters)] = local
    return probabilities


def margin_rows(probabilities: np.ndarray) -> np.ndarray:
    top = np.sort(probabilities, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def entropy_rows(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def rsp_components(
    clean_probabilities: np.ndarray,
    perturbed_probabilities: np.ndarray,
    drop_weight: float,
) -> dict[str, np.ndarray]:
    """Paper RSP: Flip plus weighted non-flip confidence-margin degradation."""
    clean = np.asarray(clean_probabilities, dtype=np.float64)
    perturbed = np.asarray(perturbed_probabilities, dtype=np.float64)
    if perturbed.ndim != 3 or perturbed.shape[1:] != clean.shape:
        raise ValueError(
            f"Expected perturbed [K,N,C] matching {clean.shape}; got {perturbed.shape}"
        )
    clean_prediction = clean.argmax(axis=1)
    noisy_prediction = perturbed.argmax(axis=2)
    flipped = noisy_prediction != clean_prediction[None, :]
    clean_margin = margin_rows(clean)
    noisy_margin = np.stack([margin_rows(value) for value in perturbed], axis=0)
    per_pass_drop = np.maximum(0.0, clean_margin[None, :] - noisy_margin)
    per_pass_drop *= ~flipped
    flip = flipped.mean(axis=0)
    drop = per_pass_drop.mean(axis=0)
    return {
        "flip": flip,
        "drop": drop,
        "rsp": flip + float(drop_weight) * drop,
    }

