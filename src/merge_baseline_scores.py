#!/usr/bin/env python
"""Merge independently computed rationale-selection score components by UID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--scores_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--components", nargs="+", default=("static", "rsp", "temp"))
    args = parser.parse_args()

    pool = read_jsonl(Path(args.data_dir) / "pool.jsonl")
    merged = {row["uid"]: {"uid": row["uid"]} for row in pool}
    component_metadata = {}
    for component in args.components:
        score_path = Path(args.scores_dir) / f"{component}_scores.jsonl"
        rows = read_jsonl(score_path)
        if len(rows) != len(pool):
            raise ValueError(f"{component}: {len(rows)} scores for {len(pool)} pool rows")
        for row in rows:
            uid = row["uid"]
            if uid not in merged:
                raise KeyError(f"Unknown UID in {score_path}: {uid}")
            overlap = (set(merged[uid]) & set(row)) - {"uid"}
            if overlap:
                raise ValueError(f"Duplicate score keys for {uid}: {sorted(overlap)}")
            merged[uid].update(row)
        metadata_path = Path(args.scores_dir) / f"{component}_metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as handle:
            component_metadata[component] = json.load(handle)

    ordered = [merged[row["uid"]] for row in pool]
    required = (
        "random",
        "complexity_ft",
        "rsr",
        "lalp",
        "aslec_drop",
        "aslec_casl",
        "rsp",
        "temp",
    )
    for index, row in enumerate(ordered):
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Merged row {index} lacks {missing}")
    output = Path(args.output)
    write_jsonl(output, ordered)
    write_json(
        output.with_name("score_metadata.json"),
        {
            "rows": len(ordered),
            "methods": list(required),
            "selection_direction": "larger is selected earlier for every method",
            "components": component_metadata,
        },
    )
    print(f"Merged {len(ordered)} rows into {output}")


if __name__ == "__main__":
    main()
