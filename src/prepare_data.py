#!/usr/bin/env python
"""Create leakage-safe fixed splits for rationale-acquisition experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset, load_dataset, load_from_disk
from huggingface_hub import HfApi, hf_hub_download

from common import LETTERS, safe_text, strip_answer_leakage, write_json, write_jsonl


def _fingerprint(rows: List[Dict[str, Any]]) -> str:
    payload = "\n".join(str(row["uid"]) for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalise_medmcqa(raw: Dataset) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, ex in enumerate(raw):
        try:
            answer_index = int(ex.get("cop", -1))
        except (TypeError, ValueError):
            continue
        options = {
            "A": safe_text(ex.get("opa")),
            "B": safe_text(ex.get("opb")),
            "C": safe_text(ex.get("opc")),
            "D": safe_text(ex.get("opd")),
        }
        rationale = strip_answer_leakage(safe_text(ex.get("exp")))
        question = safe_text(ex.get("question"))
        if answer_index not in range(4) or not question or not rationale or not all(options.values()):
            continue
        uid = safe_text(ex.get("id")) or f"medmcqa_{index}"
        rows.append(
            {
                "uid": uid,
                "question": question,
                "options": options,
                "answer_letter": LETTERS[answer_index],
                "rationale": rationale,
                "rationale_source": "MedMCQA expert/curated explanation field",
                "domain": "medical",
                "subject": safe_text(ex.get("subject_name")),
                "topic": safe_text(ex.get("topic_name")),
            }
        )
    return rows


def _pick_medprompt_json(repo_id: str) -> str:
    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    candidates = [name for name in files if name.endswith((".json", ".jsonl"))]
    if not candidates:
        raise RuntimeError(f"No JSON/JSONL data file found in {repo_id}")
    preferred = "medprompt_medqa_cot_llama31.json"
    if preferred in candidates:
        return preferred
    return sorted(candidates, key=lambda name: (("train" not in name.lower()), len(name), name))[0]


def _load_json_rows(repo_id: str, cache_dir: Optional[str]) -> List[Dict[str, Any]]:
    filename = _pick_medprompt_json(repo_id)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    with open(local_path, "r", encoding="utf-8") as handle:
        if filename.endswith(".jsonl"):
            return [json.loads(line) for line in handle if line.strip()]
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("train"), list):
            return payload["train"]
        if payload and all(isinstance(value, dict) for value in payload.values()):
            return list(payload.values())
        for value in payload.values():
            if isinstance(value, list) and all(isinstance(row, dict) for row in value):
                return value
        return [payload]
    raise TypeError(f"Unsupported dataset payload: {type(payload)}")


def _load_local_json_rows(filename: str) -> List[Dict[str, Any]]:
    path = Path(filename)
    with open(path, "r", encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("train"), list):
            return payload["train"]
        if payload and all(isinstance(value, dict) for value in payload.values()):
            return list(payload.values())
        for value in payload.values():
            if isinstance(value, list) and all(isinstance(row, dict) for row in value):
                return value
        return [payload]
    raise TypeError(f"Unsupported local dataset payload: {type(payload)}")


def _normalise_options(raw: Any) -> Optional[Dict[str, str]]:
    if isinstance(raw, dict):
        normalised = {
            letter: safe_text(raw.get(letter, raw.get(letter.lower())))
            for letter in LETTERS
            if raw.get(letter, raw.get(letter.lower())) is not None
        }
        labels = list(normalised)
        return (
            normalised
            if len(labels) in (4, 5) and labels == LETTERS[: len(labels)] and all(normalised.values())
            else None
        )
    if isinstance(raw, list) and len(raw) >= 4:
        count = min(len(raw), len(LETTERS))
        options = {letter: safe_text(raw[idx]) for idx, letter in enumerate(LETTERS[:count])}
        return options if all(options.values()) else None
    return None


def _extract_gold(ex: Dict[str, Any]) -> Optional[str]:
    for key in ("correct_answer", "answer", "result", "label"):
        value = ex.get(key)
        if value is not None:
            match = re.search(r"\b([A-D])\b", safe_text(value).upper())
            if match:
                return match.group(1)
    generations = ex.get("generations")
    if isinstance(generations, list):
        for generation in generations:
            if not isinstance(generation, dict):
                continue
            for key in ("result", "answer", "response"):
                match = re.search(
                    r"(?:final\s+)?answer\s*:\s*\[?([A-D])\]?",
                    safe_text(generation.get(key)),
                    flags=re.IGNORECASE,
                )
                if match:
                    return match.group(1).upper()
    return None


def _extract_rationale(ex: Dict[str, Any]) -> str:
    generations = ex.get("generations")
    if isinstance(generations, list):
        for generation in generations:
            if isinstance(generation, dict) and generation.get("response"):
                return strip_answer_leakage(safe_text(generation["response"]))
    for key in ("analysis", "reasoning", "cot", "response", "output", "explanation"):
        if ex.get(key):
            return strip_answer_leakage(safe_text(ex[key]))
    return ""


def _normalise_medprompt(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, ex in enumerate(raw):
        question = safe_text(ex.get("question"))
        options = _normalise_options(ex.get("options"))
        answer = _extract_gold(ex)
        rationale = _extract_rationale(ex)
        if not question or options is None or answer not in options or not rationale:
            continue
        rows.append(
            {
                "uid": safe_text(ex.get("id")) or f"medprompt_{index}",
                "question": question,
                "options": options,
                "answer_letter": answer,
                "rationale": rationale,
                "rationale_source": "Medprompt synthetic CoT",
                "domain": "medical",
                "subject": safe_text(ex.get("subject")),
                "topic": safe_text(ex.get("topic")),
            }
        )
    return rows


def _load_jsonl_source(source: str) -> List[Dict[str, Any]]:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source) as response:
            return [json.loads(line.decode("utf-8")) for line in response if line.strip()]
    with open(source, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalise_cose(
    questions: Dataset,
    explanations: List[Dict[str, Any]],
    split_name: str,
) -> List[Dict[str, Any]]:
    rationale_by_id = {}
    for ex in explanations:
        rationale = strip_answer_leakage(safe_text((ex.get("explanation") or {}).get("open-ended")))
        if rationale:
            rationale_by_id[safe_text(ex.get("id"))] = rationale

    rows: List[Dict[str, Any]] = []
    for index, ex in enumerate(questions):
        uid = safe_text(ex.get("id")) or f"cose_{split_name}_{index}"
        choice_block = ex.get("choices") or {}
        labels = [safe_text(value).upper() for value in choice_block.get("label", [])]
        texts = [safe_text(value) for value in choice_block.get("text", [])]
        options = dict(zip(labels, texts))
        answer = safe_text(ex.get("answerKey")).upper()
        question = safe_text(ex.get("question"))
        rationale = rationale_by_id.get(uid, "")
        expected = LETTERS[:5]
        if (
            labels != expected
            or len(texts) != 5
            or not all(options.get(letter) for letter in expected)
            or answer not in expected
            or not question
            or not rationale
        ):
            continue
        rows.append(
            {
                "uid": uid,
                "question": question,
                "options": {letter: options[letter] for letter in expected},
                "answer_letter": answer,
                "rationale": rationale,
                "rationale_source": "CoS-E v1.11 raw crowd open-ended explanation",
                "domain": "general",
                # TEMP expects a reusable source group. Question concepts remain
                # available as topics but are too sparse to serve as strata.
                "subject": "commonsense",
                "topic": safe_text(ex.get("question_concept")),
                "source_split": split_name,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("medmcqa", "medprompt", "cose"), required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--pool_size", type=int, default=8000)
    parser.add_argument("--dev_size", type=int, default=2000)
    parser.add_argument("--test_size", type=int, default=2000)
    parser.add_argument(
        "--medmcqa_path",
        default=(
            "./cache_directory/data/openlifescienceai___medmcqa/default/0.0.0/"
            "91c6572c454088bf71b679ad90aa8dffcd0d5868/medmcqa-train.arrow"
        ),
    )
    parser.add_argument("--medprompt_repo", default="HPAI-BSC/Medprompt-MedQA-CoT")
    parser.add_argument(
        "--medprompt_path",
        default="",
        help="Optional cached JSON/JSONL path; avoids a network lookup for MedPrompt.",
    )
    parser.add_argument("--cose_cqa_repo", default="tau/commonsense_qa")
    parser.add_argument(
        "--cose_train_explanations",
        default=(
            "https://raw.githubusercontent.com/salesforce/cos-e/refs/heads/master/"
            "data/v1.11/cose_train_v1.11.jsonl"
        ),
    )
    parser.add_argument(
        "--cose_dev_explanations",
        default=(
            "https://raw.githubusercontent.com/salesforce/cos-e/refs/heads/master/"
            "data/v1.11/cose_dev_v1.11.jsonl"
        ),
    )
    parser.add_argument("--cache_dir", default="./cache_directory/hub")
    parser.add_argument(
        "--exclude_uid_files",
        nargs="*",
        default=(),
        help="Optional JSONL split files whose UIDs must be excluded before shuffling.",
    )
    args = parser.parse_args()

    official_test_rows: Optional[List[Dict[str, Any]]] = None
    if args.dataset == "medmcqa":
        medmcqa_path = Path(args.medmcqa_path)
        if medmcqa_path.is_file() and medmcqa_path.suffix == ".arrow":
            raw = Dataset.from_file(str(medmcqa_path))
        else:
            dataset = load_from_disk(args.medmcqa_path)
            raw = dataset["train"] if isinstance(dataset, dict) or hasattr(dataset, "keys") else dataset
        rows = _normalise_medmcqa(raw)
    elif args.dataset == "medprompt":
        raw_medprompt = (
            _load_local_json_rows(args.medprompt_path)
            if args.medprompt_path
            else _load_json_rows(args.medprompt_repo, args.cache_dir)
        )
        rows = _normalise_medprompt(raw_medprompt)
    else:
        cqa = load_dataset(args.cose_cqa_repo, cache_dir=args.cache_dir)
        rows = _normalise_cose(
            cqa["train"], _load_jsonl_source(args.cose_train_explanations), "train"
        )
        official_test_rows = _normalise_cose(
            cqa["validation"], _load_jsonl_source(args.cose_dev_explanations), "validation"
        )

    excluded_uids = set()
    for filename in args.exclude_uid_files:
        with open(filename, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    excluded_uids.add(str(json.loads(line)["uid"]))
    if excluded_uids:
        rows = [row for row in rows if str(row["uid"]) not in excluded_uids]
        if official_test_rows is not None:
            official_test_rows = [
                row for row in official_test_rows if str(row["uid"]) not in excluded_uids
            ]

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if official_test_rows is None:
        requested = args.pool_size + args.dev_size + args.test_size
        if len(rows) < requested:
            raise ValueError(f"Only {len(rows)} usable rows; requested {requested}")
        pool = rows[: args.pool_size]
        dev = rows[args.pool_size : args.pool_size + args.dev_size]
        test = rows[args.pool_size + args.dev_size : requested]
    else:
        requested_train = args.pool_size + args.dev_size
        if len(rows) < requested_train:
            raise ValueError(f"Only {len(rows)} usable train rows; requested {requested_train}")
        if not official_test_rows:
            raise ValueError("CoS-E official validation split contains no usable rows")
        pool = rows[: args.pool_size]
        dev = rows[args.pool_size : requested_train]
        test = official_test_rows if args.test_size == 0 else official_test_rows[: args.test_size]

    output = Path(args.output_dir)
    write_jsonl(output / "pool.jsonl", pool)
    write_jsonl(output / "dev.jsonl", dev)
    write_jsonl(output / "test.jsonl", test)
    lengths = [len(row["rationale"].split()) for row in pool]
    metadata = {
        "dataset": args.dataset,
        "seed": args.seed,
        "available_usable_rows": len(rows),
        "available_official_test_rows": (
            len(official_test_rows) if official_test_rows is not None else None
        ),
        "excluded_uid_count": len(excluded_uids),
        "exclude_uid_files": list(args.exclude_uid_files),
        "split_sizes": {"pool": len(pool), "dev": len(dev), "test": len(test)},
        "uid_sha256": {
            "pool": _fingerprint(pool),
            "dev": _fingerprint(dev),
            "test": _fingerprint(test),
        },
        "pool_rationale_words": {
            "mean": float(np.mean(lengths)),
            "std": float(np.std(lengths)),
            "median": float(np.median(lengths)),
        },
        "acquisition_visibility": (
            "question + options + known answer + existing rationale; methods select rationales for SFT"
        ),
        "split_policy": (
            "CoS-E official train -> pool/dev; official validation -> untouched test"
            if official_test_rows is not None
            else "seeded shuffle followed by disjoint pool/dev/test slices"
        ),
    }
    write_json(output / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
