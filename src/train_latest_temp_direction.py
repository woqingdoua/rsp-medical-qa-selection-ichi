#!/usr/bin/env python
"""Train the auxiliary raw-backbone LoRA direction required by TEMP."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from common import (
    CausalCollator,
    build_lm_dataset,
    load_model,
    load_tokenizer,
    rationale_target,
    read_jsonl,
    reasoning_prompt,
    set_seed,
    write_json,
)
from original_rsp_raw_utils import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="google/medgemma-4b-it")
    parser.add_argument("--cache_dir", default="model_cache")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reference_split", default="dev", choices=("dev",))
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_rationale_chars", type=int, default=6000)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(args.seed)
    source_file = args.data_dir / f"{args.reference_split}.jsonl"
    rows = read_jsonl(source_file)
    indices = list(range(len(rows)))
    random.Random(args.seed + 2606).shuffle(indices)
    indices = indices[: min(args.samples, len(indices))]
    selected = [rows[index] for index in indices]
    tokenizer = load_tokenizer(args.model_name, args.cache_dir)
    records = [
        (
            reasoning_prompt(row),
            rationale_target(row["rationale"], row["answer_letter"], args.max_rationale_chars),
        )
        for row in selected
    ]
    train_dataset, token_stats = build_lm_dataset(records, tokenizer, args.max_length)
    model = load_model(
        args.model_name,
        args.cache_dir,
        adapter_path=None,
        trainable=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "trainer_state"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="no",
        eval_strategy="no",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        tf32=True,
        optim="adamw_torch_fused",
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    result = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=CausalCollator(tokenizer),
    ).train()
    adapter_dir = args.output_dir / "direction_adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    selected_uid_sha256 = hashlib.sha256(
        "\n".join(row["uid"] for row in selected).encode("utf-8")
    ).hexdigest()
    payload = {
        "protocol": "latest raw-MedGemma five-dataset baseline extension 2026-09-03",
        "dataset": args.dataset,
        "method_component": "TEMP auxiliary direction",
        "seed": args.seed,
        "backbone": args.model_name,
        "starting_adapter": None,
        "answer_only_warmup": False,
        "reference_split": args.reference_split,
        "reference_file": str(source_file),
        "reference_sha256": sha256(source_file),
        "reference_rationales_used": True,
        "selected_indices": indices,
        "selected_uids": [row["uid"] for row in selected],
        "selected_uid_sha256": selected_uid_sha256,
        "samples": len(selected),
        "tokenization": vars(token_stats),
        "training": {key: float(value) for key, value in result.metrics.items()},
        "adapter_dir": str(adapter_dir),
        "purpose": "Defines theta_f - theta_0 for the repository TEMP adaptation only",
        "source_reference": "Jin et al., Reasoning Quality Emerges Early, arXiv:2606.26797",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    write_json(args.output_dir / "direction_result.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
