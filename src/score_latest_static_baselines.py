#!/usr/bin/env python
"""Score Complexity-aware FT, RSR, and ASLEC-DROP from raw MedGemma.

The downstream training protocol is shared by all methods. Complexity-aware FT
is acquisition-label/rationale blind. RSR and ASLEC-DROP intrinsically score a
candidate response, so their use of the existing rationale is recorded as
privileged response visibility rather than hidden in a nominally blind protocol.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import load_model, load_tokenizer, read_jsonl, set_seed, write_json, write_jsonl
from original_rsp_raw_utils import sanitize_for_acquisition, sha256
from rationale_baselines import (
    _answer_vocab_entropy,
    _encode_steps,
    _finite_mean,
    _response_token_stats,
)


def score_rows(model, tokenizer, rows: list[dict], max_length: int, max_rationale_tokens: int,
               rank_clip: int, log_every: int) -> list[dict]:
    output = []
    for index, row in enumerate(rows):
        visible = sanitize_for_acquisition(row)
        complexity = _answer_vocab_entropy(model, tokenizer, visible, max_length)
        prompt_ids, steps = _encode_steps(tokenizer, row, max_length, max_rationale_tokens)
        stats = _response_token_stats(model, prompt_ids, steps)
        logp = stats["logp"]
        first = stats["first"]
        if len(logp):
            global_logp = float(logp.mean())
            drop_logp = _finite_mean(logp[~first], global_logp)
            raw_rsr = float(
                np.minimum(stats["rank"], rank_clip).mean()
                / max(float((-logp).mean()), 1e-8)
            )
            rsr = -raw_rsr
        else:
            global_logp = float("-inf")
            drop_logp = float("-inf")
            raw_rsr = float("inf")
            rsr = float("-inf")
        output.append(
            {
                "index": index,
                "uid": row["uid"],
                "complexity_ft": float(complexity),
                "rsr": float(rsr),
                "rsr_raw": float(raw_rsr),
                "aslec_drop": float(drop_logp),
                "rationale_mean_logp": float(global_logp),
                "rationale_tokens_scored": int(len(logp)),
                "rationale_steps": int(first.sum()) if len(first) else 0,
                "selection_unit_cost": 1,
            }
        )
        if log_every and (index + 1) % log_every == 0:
            print(f"[latest-static] {index + 1}/{len(rows)}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="google/medgemma-4b-it")
    parser.add_argument("--cache_dir", default="model_cache")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_rationale_tokens", type=int, default=384)
    parser.add_argument("--rsr_rank_clip", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--pool_limit", type=int, default=0, help="Prefix limit for smoke tests only")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(args.seed)
    started = time.time()
    rows = read_jsonl(args.data_dir / "pool.jsonl")
    if args.pool_limit:
        rows = rows[: args.pool_limit]
    tokenizer = load_tokenizer(args.model_name, args.cache_dir)
    model = load_model(args.model_name, args.cache_dir, None, trainable=False)
    scores = score_rows(
        model,
        tokenizer,
        rows,
        args.max_length,
        args.max_rationale_tokens,
        args.rsr_rank_clip,
        args.log_every,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "scores.jsonl"
    write_jsonl(score_path, scores)
    metadata = {
        "protocol": "latest raw-MedGemma five-dataset baseline extension 2026-09-03",
        "dataset": args.dataset,
        "backbone": args.model_name,
        "adapter_loaded": False,
        "scoring_initialization": "raw_backbone_without_answer_only_warmup",
        "pool_file": str(args.data_dir / "pool.jsonl"),
        "pool_sha256": sha256(args.data_dir / "pool.jsonl"),
        "rows": len(rows),
        "methods": {
            "complexity_ft": {
                "definition": "full-vocabulary entropy at the answer position",
                "selector_visible_fields": ["uid", "question", "options", "domain"],
                "rationales_used": False,
                "answers_used": False,
                "reference": "Goncharov et al., Findings of EACL 2026",
            },
            "rsr": {
                "definition": "negative rank-clipped mean-rank / mean-surprisal ratio",
                "rank_clip": args.rsr_rank_clip,
                "selector_visible_fields": ["uid", "question", "options", "domain", "rationale"],
                "rationales_used": True,
                "answers_used": False,
                "visibility_note": "privileged rationale-visible comparator required by RSR",
                "reference": "Yang et al., ACL 2026",
            },
            "aslec_drop": {
                "definition": "mean rationale-token log probability after dropping each step's first token",
                "selector_visible_fields": ["uid", "question", "options", "domain", "rationale"],
                "rationales_used": True,
                "answers_used": False,
                "visibility_note": "privileged rationale-visible comparator required by ASLEC",
                "step_segmentation": "deterministic newline and sentence heuristic",
                "reference": "Wang et al., Findings of ACL 2026",
            },
        },
        "selection_direction": "larger is selected earlier for all exported score columns",
        "score_file": str(score_path),
        "score_file_sha256": sha256(score_path),
        "elapsed_seconds": time.time() - started,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    write_json(args.output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
