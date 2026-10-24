"""CPU-only ICHI automatic-RMS RSP selection from retained components.

This release interface implements the paper's boundary rule explicitly.
Historical positive-coefficient runs are preserved separately under archive/.
No benchmark labels or auxiliary text are used to compute priorities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

EPSILON = 1e-8
SENTINEL = -1e30


def automatic_scores(flip, drop):
    if not flip or len(flip) != len(drop):
        raise ValueError("Flip and Drop must be nonempty aligned vectors")
    if any(not math.isfinite(v) or v < 0 for v in (*flip, *drop)):
        raise ValueError("Components must be finite and nonnegative")
    if any(v > 1 for v in flip):
        raise ValueError("Flip frequencies must lie in [0,1]")
    a_f = math.sqrt(math.fsum(v * v for v in flip) / len(flip))
    a_d = math.sqrt(math.fsum(v * v for v in drop) / len(drop))
    fallback = a_f <= EPSILON
    coefficient = 0.0 if fallback else a_f / (a_d + EPSILON)
    scores = list(drop) if fallback else [f + coefficient * d for f, d in zip(flip, drop)]
    return scores, {"a_F": a_f, "a_D": a_d, "lambda0": coefficient,
                    "drop_fallback": fallback, "epsilon": EPSILON}


def rank_rows(rows, budget, method="rsp_auto", full_pool=False):
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= len(rows):
        raise ValueError("Budget must be a positive integer no larger than the bank")
    if method not in ("rsp_auto", "rsp_drop", "random4b"):
        raise ValueError("Unsupported ICHI selector")
    if full_pool and method == "random4b":
        raise ValueError("Random@4B is the entropy-pool control; full-bank Random is not this reported method")
    uids = [r["uid"] for r in rows]
    if len(set(uids)) != len(uids) or any(not isinstance(u, str) or not u or "\n" in u for u in uids):
        raise ValueError("UIDs must be unique nonempty strings without line breaks")
    entropy = [float(r["entropy"]) for r in rows]
    if any(not math.isfinite(v) or v < 0 for v in entropy):
        raise ValueError("Invalid predictive entropy")
    candidates = (list(range(len(rows))) if full_pool else
                  sorted(range(len(rows)), key=lambda i: (-entropy[i], i))[:min(4 * budget, len(rows))])
    if method == "random4b":
        values = [float(rows[i]["random"]) for i in candidates]
        info = {"lambda0": None, "drop_fallback": False}
    else:
        f = [float(rows[i]["rsp_flip"]) for i in candidates]
        d = [float(rows[i]["rsp_drop"]) for i in candidates]
        values, info = automatic_scores(f, d)
        if method == "rsp_drop":
            values = d
            info = {"lambda0": None, "drop_fallback": False}
    if any(not math.isfinite(v) for v in values):
        raise ValueError("Non-finite selection score")
    priority = dict(zip(candidates, values))
    ranked = sorted(candidates, key=lambda i: (-priority[i], i))[:budget]
    selected = sorted(ranked)
    output = [{"uid": uid, method: priority.get(i, SENTINEL)} for i, uid in enumerate(uids)]
    return output, {**info, "method": method, "budget": budget, "pool_n": len(rows),
                    "candidate_n": len(candidates), "candidate_indices": candidates,
                    "coefficient_scope": "full_pool" if full_pool else "top_min_4B_N_entropy_pool",
                    "selected_indices": selected, "selected_uids": [uids[i] for i in selected],
                    "selected_uid_sha256": hashlib.sha256("\n".join(uids[i] for i in selected).encode()).hexdigest(),
                    "training_order": "ascending original pool index",
                    "tie_break": "ascending original pool index"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--budget", choices=(64, 128, 256), type=int, required=True)
    parser.add_argument("--method", choices=("rsp_auto", "rsp_drop", "random4b"), required=True)
    parser.add_argument("--full-pool", action="store_true")
    args = parser.parse_args()
    raw = args.scores.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    scores, manifest = rank_rows(rows, args.budget, args.method, args.full_pool)
    manifest["source_scores_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in scores))
    (args.output_dir / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith(("indices", "uids"))}, indent=2))


if __name__ == "__main__":
    main()
