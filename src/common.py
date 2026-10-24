#!/usr/bin/env python
"""Shared utilities for budgeted rationale-acquisition experiments.

The current rationale-visible protocol exposes questions, known answers, and
existing rationales to selectors. Legacy label-known/rationale-hidden scripts
also reuse these model, training, and evaluation helpers.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


LETTERS = ["A", "B", "C", "D", "E"]
LETTER_TO_INDEX = {letter: idx for idx, letter in enumerate(LETTERS)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def append_csv(path: str | Path, row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def strip_answer_leakage(text: str) -> str:
    """Remove explicit final-answer tails while preserving medical reasoning."""
    text = safe_text(text)
    if not text:
        return ""
    patterns = [
        r"\n+\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*\[?[A-E]\]?.*$",
        r"\n+\s*therefore[,]?\s+(?:the\s+)?(?:correct\s+)?answer\s+is\s+.*$",
        r"\n+\s*correct\s+option\s*(?:is|:)\s*\[?[A-E]\]?.*$",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def option_letters(row: Dict[str, Any]) -> List[str]:
    """Return consecutive option labels for a three-, four-, or five-choice item."""
    options = row.get("options", {})
    letters = [letter for letter in LETTERS if safe_text(options.get(letter))]
    if len(letters) not in (3, 4, 5) or letters != LETTERS[: len(letters)]:
        raise ValueError(
            f"Expected consecutive A-C/A-D/A-E options for {row.get('uid')}: {letters}"
        )
    return letters


PROMPT_TEMPLATES = ("canonical", "compact", "clinical", "options_first")


def question_block(row: Dict[str, Any], template: str | None = None) -> str:
    template = template or safe_text(row.get("_prompt_template")) or "canonical"
    if template not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt template: {template}")
    qualifier = " medical" if row.get("domain") == "medical" else ""
    options = "".join(
        f"{letter}. {row['options'][letter].strip()}\n" for letter in option_letters(row)
    )
    question = row["question"].strip()
    if template == "canonical":
        return f"Please answer the following multiple-choice{qualifier} question:\n{question}\n" + options
    if template == "compact":
        return f"Q: {question}\nChoices:\n" + options
    if template == "clinical":
        return f"Clinical multiple-choice case:\nQuestion: {question}\nCandidate answers:\n" + options
    return "Candidate answers:\n" + options + f"Question:\n{question}\n"


def answer_prompt(row: Dict[str, Any]) -> str:
    template = safe_text(row.get("_prompt_template")) or "canonical"
    if template == "canonical":
        instruction = "\nChoose one option. Return only the final option letter.\nFinal Answer:"
    elif template == "compact":
        instruction = "\nReply with one choice letter only.\nAnswer:"
    elif template == "clinical":
        instruction = "\nSelect the single best answer. Output only its letter.\nSelected answer:"
    elif template == "options_first":
        instruction = "\nWhich option best answers the question? Return only the letter.\nChoice:"
    else:
        raise ValueError(f"Unknown prompt template: {template}")
    return question_block(row, template) + instruction


def reasoning_prompt(row: Dict[str, Any]) -> str:
    return (
        question_block(row)
        + "\nThink through the evidence and relevant knowledge step by step. "
        + "End with `Final Answer: <LETTER>`.\n"
    )


def explanation_prompt(row: Dict[str, Any], predicted_letter: str) -> str:
    return (
        question_block(row)
        + f"\nA model predicted option {predicted_letter}. Explain the reasoning "
        + "that would justify this prediction. Do not merely repeat the option text.\n"
        + "Explanation:"
    )


def answer_target(letter: str) -> str:
    return f" {letter}\n"


def rationale_target(rationale: str, letter: str, max_chars: int) -> str:
    rationale = strip_answer_leakage(rationale)
    if max_chars > 0:
        rationale = rationale[:max_chars].rstrip()
    return f" {rationale}\nFinal Answer: {letter}\n"


@dataclass
class TokenizedStats:
    total: int
    kept: int
    dropped_prompt_too_long: int
    truncated_targets: int


def build_lm_dataset(
    records: Sequence[Tuple[str, str]],
    tokenizer,
    max_length: int,
    min_target_tail_tokens: int = 8,
) -> Tuple[Dataset, TokenizedStats]:
    """Tokenize prompt/target pairs, masking prompt tokens from the loss.

    If a rationale is too long, its prefix is truncated but the final answer tail
    remains supervised. This prevents the silent answer truncation in the legacy
    pipeline.
    """
    rows: List[Dict[str, Any]] = []
    dropped = 0
    truncated = 0
    eos_id = tokenizer.eos_token_id

    for prompt, target in records:
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        t_ids = tokenizer.encode(target, add_special_tokens=False)
        if eos_id is not None and (not t_ids or t_ids[-1] != eos_id):
            t_ids = t_ids + [eos_id]

        if len(p_ids) + min_target_tail_tokens >= max_length:
            dropped += 1
            continue

        available = max_length - len(p_ids)
        if len(t_ids) > available:
            truncated += 1
            tail_len = min(min_target_tail_tokens, available)
            prefix_len = max(0, available - tail_len)
            t_ids = t_ids[:prefix_len] + t_ids[-tail_len:]

        ids = p_ids + t_ids
        labels = [-100] * len(p_ids) + t_ids
        rows.append(
            {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "labels": labels,
            }
        )

    stats = TokenizedStats(
        total=len(records),
        kept=len(rows),
        dropped_prompt_too_long=dropped,
        truncated_targets=truncated,
    )
    return Dataset.from_list(rows), stats


class CausalCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in features)
        input_ids, masks, labels = [], [], []
        for item in features:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_id] * pad)
            masks.append(item["attention_mask"] + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_tokenizer(model_name: str, cache_dir: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(
    model_name: str,
    cache_dir: Optional[str],
    adapter_path: Optional[str],
    trainable: bool,
    lora_rank: int = 16,
    lora_alpha: int = 32,
):
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
    elif trainable:
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    if trainable:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
    else:
        model.config.use_cache = True
    if torch.cuda.is_available():
        model.to("cuda")
    return model


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


@torch.no_grad()
def conditional_logp(
    model,
    tokenizer,
    prompts: Sequence[str],
    completions: Sequence[str],
    max_length: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    sequences: List[List[int]] = []
    supervised: List[List[int]] = []
    for prompt, completion in zip(prompts, completions):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)
        available_prompt = max_length - len(completion_ids)
        if available_prompt <= 0:
            raise ValueError("Completion alone exceeds max_length")
        if len(prompt_ids) > available_prompt:
            # Preserve both the question header and the final options/instruction.
            head = available_prompt // 2
            prompt_ids = prompt_ids[:head] + prompt_ids[-(available_prompt - head) :]
        ids = prompt_ids + completion_ids
        sequences.append(ids)
        supervised.append([-100] * len(prompt_ids) + completion_ids)

    max_batch_length = max(len(ids) for ids in sequences)
    padded_ids, padded_masks, padded_labels = [], [], []
    for ids, labels in zip(sequences, supervised):
        pad = max_batch_length - len(ids)
        padded_ids.append([tokenizer.pad_token_id] * pad + ids)
        padded_masks.append([0] * pad + [1] * len(ids))
        padded_labels.append([-100] * pad + labels)
    input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(padded_masks, dtype=torch.long, device=device)
    labels_tensor = torch.tensor(padded_labels, dtype=torch.long, device=device)

    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits[:, :-1, :]
    labels = labels_tensor[:, 1:]
    mask = (labels != -100).float()
    log_probs = F.log_softmax(logits, dim=-1)
    safe_labels = labels.masked_fill(labels == -100, 0)
    token_logp = torch.gather(log_probs, -1, safe_labels.unsqueeze(-1)).squeeze(-1)
    result = (token_logp * mask).sum(-1)
    return result.float().cpu().numpy()


@torch.no_grad()
def candidate_probabilities(
    model,
    tokenizer,
    rows: Sequence[Dict[str, Any]],
    batch_size: int,
    max_length: int,
    set_eval: bool = True,
) -> np.ndarray:
    if set_eval:
        model.eval()
    if not rows:
        return np.empty((0, 0), dtype=np.float64)
    row_letters = [tuple(option_letters(row)) for row in rows]
    max_choices = max(len(letters) for letters in row_letters)
    output = np.zeros((len(rows), max_choices), dtype=np.float64)
    # Some medical datasets mix four- and five-choice questions.  Evaluate
    # identical option vocabularies together, then leave unavailable trailing
    # choices at probability zero in the common output matrix.
    groups: Dict[Tuple[str, ...], List[int]] = {}
    for index, letters in enumerate(row_letters):
        groups.setdefault(letters, []).append(index)
    for letters, indices in groups.items():
        for start in range(0, len(indices), batch_size):
            local_indices = indices[start : start + batch_size]
            batch = [rows[index] for index in local_indices]
            prompts = [answer_prompt(row) for row in batch]
            scores = []
            for letter in letters:
                scores.append(
                    conditional_logp(
                        model,
                        tokenizer,
                        prompts,
                        [answer_target(letter)] * len(batch),
                        max_length,
                    )
                )
            probs = softmax_rows(np.stack(scores, axis=1).astype(np.float64))
            output[np.asarray(local_indices), : len(letters)] = probs
    return output


def entropy_rows(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-12, 1.0)
    return -(probs * np.log(probs)).sum(axis=1)


def margin_rows(probs: np.ndarray) -> np.ndarray:
    top = np.sort(probs, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def classification_metrics(probs: np.ndarray, gold: np.ndarray) -> Dict[str, float]:
    pred = probs.argmax(axis=1)
    accuracy = float(np.mean(pred == gold))
    f1_values = []
    for label in range(probs.shape[1]):
        tp = int(np.sum((pred == label) & (gold == label)))
        fp = int(np.sum((pred == label) & (gold != label)))
        fn = int(np.sum((pred != label) & (gold == label)))
        denominator = 2 * tp + fp + fn
        f1_values.append(0.0 if denominator == 0 else 2 * tp / denominator)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(gold)), gold] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    confidence = probs.max(axis=1)
    correct = (pred == gold).astype(np.float64)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for idx in range(15):
        lo, hi = edges[idx], edges[idx + 1]
        chosen = (confidence >= lo) & ((confidence <= hi) if idx == 14 else (confidence < hi))
        if chosen.any():
            ece += float(chosen.mean()) * abs(float(correct[chosen].mean()) - float(confidence[chosen].mean()))
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "brier": brier,
        "ece": float(ece),
    }


def evaluate_rows(
    model,
    tokenizer,
    rows: Sequence[Dict[str, Any]],
    batch_size: int,
    max_length: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    probs = candidate_probabilities(model, tokenizer, rows, batch_size, max_length)
    gold = np.array([LETTER_TO_INDEX[row["answer_letter"]] for row in rows], dtype=np.int64)
    return classification_metrics(probs, gold), probs, gold


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    std = values.std()
    if std < 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def get_transformer_blocks(model):
    candidates = [
        lambda m: m.layers,
        lambda m: m.model.layers,
        lambda m: m.model.language_model.layers,
        lambda m: m.model.model.layers,
        lambda m: m.model.model.language_model.layers,
        lambda m: m.base_model.model.model.layers,
        lambda m: m.base_model.model.model.language_model.layers,
        lambda m: m.transformer.h,
        lambda m: m.gpt_neox.layers,
    ]
    queue = [model]
    seen = set()
    while queue:
        current = queue.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        for candidate in candidates:
            try:
                blocks = candidate(current)
                if blocks is not None and len(blocks) > 0:
                    return blocks
            except Exception:
                pass
        for attr in ("module", "base_model", "model", "language_model"):
            child = getattr(current, attr, None)
            if child is not None and child is not current:
                queue.append(child)
    raise RuntimeError("Could not locate transformer blocks for hidden-state perturbation")


def hidden_noise_hook(sigma: float, seed: int, relative: bool = False):
    def hook(_module, _inputs, output):
        def perturb(tensor: torch.Tensor) -> torch.Tensor:
            generator = torch.Generator(device=tensor.device)
            generator.manual_seed(seed)
            scale: torch.Tensor | float = sigma
            if relative:
                dims = tuple(range(1, tensor.ndim))
                rms = tensor.float().pow(2).mean(dim=dims, keepdim=True).sqrt().to(tensor.dtype)
                scale = sigma * rms
            noise = torch.randn(
                tensor.shape,
                generator=generator,
                device=tensor.device,
                dtype=tensor.dtype,
            )
            return tensor + noise * scale

        if isinstance(output, torch.Tensor):
            return perturb(output)
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return (perturb(output[0]),) + output[1:]
        if isinstance(output, list) and output and isinstance(output[0], torch.Tensor):
            return [perturb(output[0])] + output[1:]
        return output

    return hook


def parse_final_answer(text: str) -> str:
    matches = re.findall(r"final\s+answer\s*:\s*\[?([A-E])\]?", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    matches = re.findall(r"(?:answer|option)\s+(?:is\s+)?\[?([A-E])\]?", text, flags=re.IGNORECASE)
    return matches[-1].upper() if matches else "?"


def empirical_answer_entropy(answers: Sequence[str]) -> float:
    counts: Dict[str, int] = {}
    for answer in answers:
        counts[answer] = counts.get(answer, 0) + 1
    total = max(1, len(answers))
    return float(-sum((count / total) * math.log(count / total) for count in counts.values()))
