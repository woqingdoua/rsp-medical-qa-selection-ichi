#!/usr/bin/env python
"""Train a paper-style CoT-as-context branch from raw MedGemma."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from common import (
    CausalCollator,
    LETTER_TO_INDEX,
    answer_target,
    classification_metrics,
    load_model,
    load_tokenizer,
    question_block,
    read_jsonl,
    set_seed,
    strip_answer_leakage,
    write_json,
)
from original_rsp_raw_utils import next_letter_probabilities, sha256


def select_exact(
    pool: list[dict], scores: list[dict], method: str, budget: int,
    selection_order: str = "score_rank",
) -> list[int]:
    if len(pool) != len(scores):
        raise ValueError(f"Pool/score length mismatch: {len(pool)} != {len(scores)}")
    for index, (row, score) in enumerate(zip(pool, scores)):
        if row["uid"] != score["uid"]:
            raise ValueError(f"UID mismatch at {index}: {row['uid']} != {score['uid']}")
    ordered = sorted(
        range(len(pool)), key=lambda index: (-float(scores[index][method]), index)
    )
    chosen = ordered[:budget]
    if len(chosen) != budget:
        raise RuntimeError(f"Selected {len(chosen)} rows, expected {budget}")
    if method.startswith("rsp") and any(
        float(scores[index][method]) <= -1e29 for index in chosen
    ):
        raise RuntimeError("RSP selection escaped the entropy-prefilter candidate set")
    if selection_order == "pool_index":
        chosen = sorted(chosen)
    elif selection_order != "score_rank":
        raise ValueError(f"Unknown selection order: {selection_order}")
    return chosen


def build_final_answer_only_dataset(rows: list[dict], tokenizer, max_length: int):
    """Tokenize CoT as context while supervising only the final answer target."""
    examples = []
    truncated_rationales = 0
    truncated_questions = 0
    eos_id = tokenizer.eos_token_id
    for row in rows:
        prefix_ids = tokenizer.encode(
            question_block(row) + "\nReasoning:\n", add_special_tokens=False
        )
        rationale_ids = tokenizer.encode(
            strip_answer_leakage(str(row["rationale"])), add_special_tokens=False
        )
        suffix_ids = tokenizer.encode(
            "\nUsing the reasoning above, return only the final option letter.\nFinal Answer:",
            add_special_tokens=False,
        )
        target_ids = tokenizer.encode(
            answer_target(row["answer_letter"]), add_special_tokens=False
        )
        if eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
            target_ids.append(eos_id)
        prompt_budget = max_length - len(target_ids)
        minimum_rationale = min(32, len(rationale_ids))
        if len(prefix_ids) + len(suffix_ids) + minimum_rationale > prompt_budget:
            keep = prompt_budget - len(suffix_ids) - minimum_rationale
            if keep <= 0:
                raise ValueError(f"Instruction and target exceed max length for {row['uid']}")
            head = keep // 2
            prefix_ids = prefix_ids[:head] + prefix_ids[-(keep - head) :]
            truncated_questions += 1
        available_rationale = prompt_budget - len(prefix_ids) - len(suffix_ids)
        if len(rationale_ids) > available_rationale:
            rationale_ids = rationale_ids[:available_rationale]
            truncated_rationales += 1
        prompt_ids = prefix_ids + rationale_ids + suffix_ids
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        if len(input_ids) > max_length:
            raise RuntimeError(f"Tokenization overflow for {row['uid']}")
        examples.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
        )
    return Dataset.from_list(examples), {
        "total": len(rows),
        "kept": len(examples),
        "truncated_rationales": truncated_rationales,
        "truncated_questions": truncated_questions,
        "loss_scope": "final-answer target tokens only",
    }


@torch.no_grad()
def evaluate(model, tokenizer, rows: list[dict], batch_size: int, max_length: int):
    probabilities = next_letter_probabilities(
        model, tokenizer, rows, batch_size, max_length
    )
    gold = np.asarray([LETTER_TO_INDEX[row["answer_letter"]] for row in rows], dtype=np.int64)
    return classification_metrics(probabilities, gold), probabilities, gold


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--score_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--method",
        required=True,
        help="Score column to rank; supports derived rsp_lambda_* columns.",
    )
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--model_name", default="google/medgemma-4b-it")
    parser.add_argument("--cache_dir", default="model_cache")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument(
        "--skip_adapter_save",
        action="store_true",
        help="Evaluate the in-memory adapter without persisting reconstructible weights.",
    )
    parser.add_argument(
        "--eval_splits",
        nargs="+",
        choices=("dev", "test"),
        default=("dev", "test"),
        help="Use dev-only for hyperparameter candidates so test remains unopened.",
    )
    parser.add_argument(
        "--selection_order",
        choices=("score_rank", "pool_index"),
        default="score_rank",
        help="Order the selected set by score rank or canonical source-pool index before training.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    pool = read_jsonl(args.data_dir / "pool.jsonl")
    evaluation_rows = {
        split: read_jsonl(args.data_dir / f"{split}.jsonl")
        for split in args.eval_splits
    }
    scores = read_jsonl(args.score_file)
    selected_indices = select_exact(
        pool, scores, args.method, args.budget, args.selection_order
    )
    selected = [pool[index] for index in selected_indices]
    tokenizer = load_tokenizer(args.model_name, args.cache_dir)
    train_dataset, token_stats = build_final_answer_only_dataset(
        selected, tokenizer, args.max_length
    )
    if len(train_dataset) != args.budget:
        raise RuntimeError(
            f"Expected {args.budget} training records after tokenization; got {len(train_dataset)}"
        )
    model = load_model(
        args.model_name,
        args.cache_dir,
        adapter_path=None,
        trainable=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "trainer_state"),
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=25,
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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=CausalCollator(tokenizer),
    )
    train_result = trainer.train()
    adapter_dir = args.output_dir / "adapter"
    if not args.skip_adapter_save:
        model.save_pretrained(adapter_dir)
    del trainer
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    model.eval()
    split_metrics = {}
    prediction_arrays = {}
    for split in args.eval_splits:
        metrics, probabilities, gold = evaluate(
            model,
            tokenizer,
            evaluation_rows[split],
            args.eval_batch_size,
            args.max_length,
        )
        split_metrics[split] = metrics
        prediction_arrays[f"{split}_probs"] = probabilities
        prediction_arrays[f"{split}_gold"] = gold
    np.savez_compressed(args.output_dir / "predictions.npz", **prediction_arrays)
    result = {
        "protocol": "Original RSP--Raw",
        "dataset": args.dataset,
        "method": args.method,
        "budget_items": args.budget,
        "seed": args.seed,
        "backbone": args.model_name,
        "training_initialization": "raw_backbone_new_lora_without_answer_only_warmup",
        "starting_adapter": None,
        "adapter_saved": not args.skip_adapter_save,
        "rationale_role": "input_context_only",
        "supervised_target": "final_answer_only",
        "rationale_tokens_in_loss": False,
        "answer_labels_visible_to_selector": False,
        "selection_training_order": args.selection_order,
        "selected_uid_order": [row["uid"] for row in selected],
        "selected_uid_sha256": __import__("hashlib").sha256(
            "\n".join(row["uid"] for row in selected).encode()
        ).hexdigest(),
        "score_file": str(args.score_file),
        "score_file_sha256": sha256(args.score_file),
        "tokenization": token_stats,
        "training": {key: float(value) for key, value in train_result.metrics.items()},
        "evaluation_scope": list(args.eval_splits),
        **split_metrics,
        "elapsed_seconds": time.time() - started,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    write_json(args.output_dir / "result.json", result)
    summary_keys = ("dataset", "method", "seed", *args.eval_splits)
    print(json.dumps({key: result[key] for key in summary_keys}, indent=2))


if __name__ == "__main__":
    main()
