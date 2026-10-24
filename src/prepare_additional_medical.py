#!/usr/bin/env python
"""Prepare fixed splits for four additional medical rationale datasets.

The normalised schema matches the existing rationale-acquisition pipeline:
uid, question, consecutive options, answer_letter, rationale, and source
metadata.  AfriMed-QA and MedExpQA retain their official test splits;
MedExQA and PubMedQA receive deterministic derived splits because they do not
provide a training/pool split suitable for acquisition simulation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from datasets import load_dataset

from common import LETTERS, safe_text, strip_answer_leakage, write_json, write_jsonl


DATASET_VERSIONS = {
    "afrimedqa": "intron-innovation/AfriMed-QA version-1 tag, commit 8c04403d2cc613e43d72c6468251a2d42409d397",
    "medexpqa": "HiTZ/MedExpQA, English config, repository commit fa92fb621fdac8ce07a99928b2a599a14541a28d",
    "medexqa": "bluesky333/MedExQA, commit 3a9f80d6de2c354956d7e86d323214b92d547d6e",
    "pubmedqa": "qiaojin/PubMedQA pqa_labeled",
}


def _fingerprint(rows: Sequence[dict]) -> str:
    return hashlib.sha256("\n".join(str(row["uid"]) for row in rows).encode()).hexdigest()


def _normalised_question(text: str) -> str:
    return re.sub(r"\s+", " ", safe_text(text)).strip().lower()


def _deduplicate(rows: Iterable[dict]) -> Tuple[List[dict], int]:
    output, seen = [], set()
    duplicates = 0
    for row in rows:
        key = _normalised_question(row["question"])
        if not key or key in seen:
            duplicates += 1
            continue
        seen.add(key)
        output.append(row)
    return output, duplicates


def _shuffle(rows: Sequence[dict], seed: int) -> List[dict]:
    output = list(rows)
    random.Random(seed).shuffle(output)
    return output


def _derived_split(
    rows: Sequence[dict], pool_size: int, dev_size: int, test_size: int, seed: int
) -> Tuple[List[dict], List[dict], List[dict]]:
    rows = _shuffle(rows, seed)
    requested = pool_size + dev_size + test_size
    if len(rows) < requested:
        raise ValueError(f"Only {len(rows)} rows available; requested {requested}")
    return (
        rows[:pool_size],
        rows[pool_size : pool_size + dev_size],
        rows[pool_size + dev_size : requested],
    )


def _parse_afrimed_options(value: str) -> Dict[str, str] | None:
    try:
        raw = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    options = {
        letter: safe_text(raw.get(f"option{index}"))
        for index, letter in enumerate(LETTERS, start=1)
    }
    return options if all(options.values()) else None


def _load_afrimedqa(path: str, seed: int) -> Tuple[List[dict], List[dict], List[dict], dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = []
    rejected = Counter()
    for index, ex in enumerate(raw_rows):
        if safe_text(ex.get("question_type")).lower() != "mcq":
            continue
        options = _parse_afrimed_options(ex.get("answer_options", ""))
        match = re.fullmatch(r"option([1-5])", safe_text(ex.get("correct_answer")).lower())
        question = safe_text(ex.get("question"))
        rationale = strip_answer_leakage(safe_text(ex.get("answer_rationale")))
        if options is None:
            rejected["invalid_options"] += 1
            continue
        if match is None:
            rejected["not_single_answer"] += 1
            continue
        if not question or not rationale:
            rejected["missing_question_or_rationale"] += 1
            continue
        rows.append(
            {
                "uid": safe_text(ex.get("sample_id")) or f"afrimedqa_{index}",
                "question": question,
                "options": options,
                "answer_letter": LETTERS[int(match.group(1)) - 1],
                "rationale": rationale,
                "rationale_source": "AfriMed-QA v1 human-authored answer_rationale",
                "domain": "medical",
                "subject": safe_text(ex.get("specialty")) or "unknown",
                "topic": safe_text(ex.get("question_source")),
                "country": safe_text(ex.get("country")),
                "source_split": safe_text(ex.get("split")),
            }
        )
    rows, duplicates = _deduplicate(rows)
    train = [row for row in rows if row["source_split"] == "train"]
    test = [row for row in rows if row["source_split"] == "test"]
    train = _shuffle(train, seed)
    # Use all official training questions: 1,500 for the acquisition pool and
    # the remainder for development.  The official test remains untouched.
    pool = train[: min(1500, len(train))]
    dev = train[len(pool) :]
    if len(pool) < 1000 or len(dev) < 300 or len(test) < 500:
        raise ValueError(
            f"Unexpected AfriMed-QA sizes after filtering: {len(pool)}/{len(dev)}/{len(test)}"
        )
    return pool, dev, test, {"rejected": dict(rejected), "duplicates_removed": duplicates}


def _medexp_rationale(ex: dict) -> str:
    # The dataset-provided no-ref field masks explicit correct-option cues while
    # preserving the doctor's clinical explanation and distractor discussion.
    text = safe_text(ex.get("full_answer_no_ref") or ex.get("full_answer"))
    text = re.sub(r"\s*\[HIDDEN\]\s*", " ", text)
    return strip_answer_leakage(re.sub(r"\s+", " ", text).strip())


def _normalise_medexp_split(rows: Iterable[dict], split: str) -> List[dict]:
    output = []
    for index, ex in enumerate(rows):
        raw_options = ex.get("options") or {}
        option_values = [safe_text(raw_options.get(str(number))) for number in range(1, 6)]
        while option_values and not option_values[-1]:
            option_values.pop()
        options = {
            letter: value for letter, value in zip(LETTERS[: len(option_values)], option_values)
        }
        try:
            answer_index = int(ex.get("correct_option")) - 1
        except (TypeError, ValueError):
            continue
        rationale = _medexp_rationale(ex)
        question = safe_text(ex.get("full_question"))
        if (
            len(options) not in (4, 5)
            or answer_index not in range(len(options))
            or not all(options.values())
            or not question
            or not rationale
        ):
            continue
        output.append(
            {
                "uid": f"medexpqa_en_{safe_text(ex.get('id')) or f'{split}_{index}'}",
                "question": question,
                "options": options,
                "answer_letter": LETTERS[answer_index],
                "rationale": rationale,
                "rationale_source": "MedExpQA/CasiMedicos doctor-authored full_answer_no_ref",
                "domain": "medical",
                "subject": safe_text(ex.get("type")) or "unknown",
                "topic": f"MIR-{safe_text(ex.get('year'))}",
                "source_split": split,
            }
        )
    return output


def _load_medexpqa(cache_dir: str) -> Tuple[List[dict], List[dict], List[dict], dict]:
    dataset = load_dataset("HiTZ/MedExpQA", "en", cache_dir=cache_dir)
    pool = _normalise_medexp_split(dataset["train"], "train")
    dev = _normalise_medexp_split(dataset["validation"], "validation")
    test = _normalise_medexp_split(dataset["test"], "test")
    return pool, dev, test, {"official_splits": True}


def _load_medexqa(root: str, seed: int) -> Tuple[List[dict], List[dict], List[dict], dict]:
    rows = []
    for path in sorted(Path(root).glob("*/*.tsv")):
        specialty = path.stem.rsplit("_", 1)[0]
        source_split = path.parent.name
        with open(path, "r", encoding="utf-8", newline="") as handle:
            for index, values in enumerate(csv.reader(handle, delimiter="\t")):
                if len(values) != 8:
                    continue
                question, *rest = values
                options = {letter: safe_text(rest[i]) for i, letter in enumerate(LETTERS[:4])}
                rationale = strip_answer_leakage(safe_text(rest[4]))
                rationale_alt = strip_answer_leakage(safe_text(rest[5]))
                answer = safe_text(rest[6]).upper()
                if not question or not all(options.values()) or answer not in options or not rationale:
                    continue
                uid_payload = f"{specialty}|{source_split}|{index}|{question}".encode()
                uid = "medexqa_" + hashlib.sha256(uid_payload).hexdigest()[:20]
                rows.append(
                    {
                        "uid": uid,
                        "question": safe_text(question),
                        "options": options,
                        "answer_letter": answer,
                        "rationale": rationale,
                        "rationale_alt": rationale_alt,
                        "rationale_source": "MedExQA explanation pair, first reference",
                        "domain": "medical",
                        "subject": specialty,
                        "topic": specialty,
                        "source_split": source_split,
                    }
                )
    rows, duplicates = _deduplicate(rows)
    pool, dev, test = _derived_split(rows, 600, 164, 200, seed)
    return pool, dev, test, {
        "official_splits": False,
        "derived_split_reason": "MedExQA has 25 few-shot dev and 940 benchmark test rows but no acquisition-training split",
        "duplicates_removed": duplicates,
    }


def _load_pubmedqa(cache_dir: str, seed: int) -> Tuple[List[dict], List[dict], List[dict], dict]:
    dataset = load_dataset(
        "qiaojin/PubMedQA", "pqa_labeled", cache_dir=cache_dir, trust_remote_code=True
    )["train"]
    label_map = {"yes": "A", "no": "B", "maybe": "C"}
    rows = []
    for index, ex in enumerate(dataset):
        label = safe_text(ex.get("final_decision")).lower()
        context = ex.get("context") or {}
        contexts = [safe_text(value) for value in context.get("contexts", []) if safe_text(value)]
        question = safe_text(ex.get("question"))
        rationale = strip_answer_leakage(safe_text(ex.get("long_answer")))
        if label not in label_map or not contexts or not question or not rationale:
            continue
        combined_question = "Biomedical abstract:\n" + "\n".join(contexts) + "\nResearch question: " + question
        rows.append(
            {
                "uid": f"pubmedqa_{safe_text(ex.get('pubid')) or index}",
                "question": combined_question,
                "options": {"A": "Yes", "B": "No", "C": "Maybe"},
                "answer_letter": label_map[label],
                "rationale": rationale,
                "rationale_source": "PubMedQA article-abstract conclusion (long_answer)",
                "domain": "medical",
                "subject": "biomedical_research",
                "topic": "; ".join(safe_text(value) for value in context.get("meshes", [])[:5]),
                "source_split": "pqa_labeled",
            }
        )
    rows, duplicates = _deduplicate(rows)
    pool, dev, test = _derived_split(rows, 600, 200, 200, seed)
    return pool, dev, test, {
        "official_splits": False,
        "derived_split_reason": "PQA-L contains one 1,000-row expert-labelled split",
        "duplicates_removed": duplicates,
    }


def _label_counts(rows: Sequence[dict]) -> dict:
    return dict(sorted(Counter(row["answer_letter"] for row in rows).items()))


def _write_dataset(output_root: str, name: str, splits, extra: dict, seed: int) -> None:
    pool, dev, test = splits
    output = Path(output_root) / name
    write_jsonl(output / "pool.jsonl", pool)
    write_jsonl(output / "dev.jsonl", dev)
    write_jsonl(output / "test.jsonl", test)
    lengths = [len(row["rationale"].split()) for row in pool]
    metadata = {
        "dataset": name,
        "source_version": DATASET_VERSIONS[name],
        "seed": seed,
        "split_sizes": {"pool": len(pool), "dev": len(dev), "test": len(test)},
        "uid_sha256": {
            "pool": _fingerprint(pool),
            "dev": _fingerprint(dev),
            "test": _fingerprint(test),
        },
        "label_counts": {
            "pool": _label_counts(pool),
            "dev": _label_counts(dev),
            "test": _label_counts(test),
        },
        "pool_rationale_words": {
            "mean": float(np.mean(lengths)),
            "std": float(np.std(lengths)),
            "median": float(np.median(lengths)),
        },
        "acquisition_visibility": "question + options + known answer + existing rationale",
        **extra,
    }
    write_json(output / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_VERSIONS),
        default=tuple(DATASET_VERSIONS),
    )
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--afrimed_path",
        default=(
            "./external/AfriMed-QA-v1/"
            "data/afri_med_qa_10k_v1_1_phase_1.csv"
        ),
    )
    parser.add_argument(
        "--medexqa_root",
        default="./external/MedExQA",
    )
    parser.add_argument("--cache_dir", default="./cache_directory/hub")
    args = parser.parse_args()

    loaders: Dict[str, Callable[[], tuple]] = {
        "afrimedqa": lambda: _load_afrimedqa(args.afrimed_path, args.seed),
        "medexpqa": lambda: _load_medexpqa(args.cache_dir),
        "medexqa": lambda: _load_medexqa(args.medexqa_root, args.seed),
        "pubmedqa": lambda: _load_pubmedqa(args.cache_dir, args.seed),
    }
    for name in args.datasets:
        pool, dev, test, extra = loaders[name]()
        _write_dataset(args.output_root, name, (pool, dev, test), extra, args.seed)


if __name__ == "__main__":
    main()
