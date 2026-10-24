"""Deterministic summaries of already evaluated ICHI result rows; no inference.

The retained CSVs contain unrounded per-run metrics, not per-item predictions.
This reproduces aggregate means/sample SDs, not a new checkpoint evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATASETS = ("afrimedqa", "medexpqa", "medexqa", "pubmedqa", "medmcqa")
SEEDS = (13, 23, 37)


def read(name):
    with (RESULTS / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def normalized(rows, default_budget=None, rename=None):
    output = {}
    for r in rows:
        key = (r["dataset"], int(r.get("budget", default_budget)), int(r["seed"]),
               (rename or {}).get(r["method"], r["method"]))
        if key in output:
            raise ValueError(f"Duplicate run {key}")
        output[key] = (float(r.get("accuracy", r.get("test_accuracy"))),
                       float(r.get("macro_f1", r.get("test_macro_f1"))))
    return output


def summary(values):
    rows = []
    for budget in sorted({key[1] for key in values}):
        for method in sorted({key[3] for key in values if key[1] == budget}):
            available = [ds for ds in DATASETS if all((ds, budget, s, method) in values for s in SEEDS)]
            for ds in (*available, "dataset_macro"):
                per_seed = [tuple(st.mean(values[(d, budget, s, method)][m] for d in available)
                                  for m in (0, 1)) if ds == "dataset_macro" else values[(ds, budget, s, method)]
                            for s in SEEDS]
                rows.append({"dataset": ds, "budget": budget, "method": method, "n_seeds": 3,
                    "accuracy_mean": st.mean(v[0] for v in per_seed),
                    "accuracy_sample_sd": st.stdev(v[0] for v in per_seed),
                    "macro_f1_mean": st.mean(v[1] for v in per_seed),
                    "macro_f1_sample_sd": st.stdev(v[1] for v in per_seed)})
    return rows


def csv_text(rows):
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true", help="Write deterministic derived summaries only")
    a = p.parse_args()
    budget = normalized(read("rsp_random_budget_per_seed.csv"), rename={"rsp": "rsp_auto"})
    drop = normalized(read("drop_budget_per_seed.csv"), rename={"preentropy_drop_only": "rsp_drop"})
    assert len(budget) == 90 and len(drop) == 45
    budget.update(drop)
    expected = {(d, b, s, m) for d in DATASETS for b in (64, 128, 256) for s in SEEDS
                for m in ("rsp_auto", "rsp_drop", "random4b")}
    assert set(budget) == expected
    primary = normalized(read("baselines_b128_per_seed.csv"), 128,
                         {"entropy_prefiltered_random": "random4b"})
    for k, v in budget.items():
        if k[1] == 128:
            if k in primary:
                assert primary[k] == v, f"Canonical baseline/budget disagreement: {k}"
            primary[k] = v
    assert len(primary) == 195
    summaries = {"primary_summary.csv": summary(primary), "budget_summary.csv": summary(budget)}
    target_f1 = {"rsp_auto": 47.55, "rsp_drop": 48.28, "random4b": 45.11}
    for method, target in target_f1.items():
        row = next(r for r in summaries["primary_summary.csv"] if r["dataset"] == "dataset_macro" and r["method"] == method)
        assert round(row["macro_f1_mean"] * 100, 2) == target
    full_rsp = normalized(read("fullpool_rsp_per_seed.csv"), rename={"full_pool_rsp_auto": "rsp_auto"})
    full_drop = normalized(read("fullpool_drop_per_seed.csv"), rename={"full_pool_drop_only_current": "rsp_drop"})
    assert len(full_rsp) == 27 and len(full_drop) == 45
    sizes = {"afrimedqa": 1500, "medexpqa": 434, "medexqa": 600, "pubmedqa": 600, "medmcqa": 512}
    paired = []
    for key, full in {**full_rsp, **full_drop}.items():
        ds, b, seed, method = key
        if min(4 * b, sizes[ds]) == sizes[ds]:
            continue
        pre = budget[key]
        paired.append({"dataset": ds, "budget": b, "seed": seed, "method": method,
                       "prefilter_accuracy": pre[0], "fullpool_accuracy": full[0],
                       "accuracy_delta": pre[0] - full[0], "prefilter_macro_f1": pre[1],
                       "fullpool_macro_f1": full[1], "macro_f1_delta": pre[1] - full[1]})
    assert len(paired) == 54
    summaries["entropy_ablation_paired.csv"] = sorted(paired, key=lambda r: (r["budget"], r["dataset"], r["method"], r["seed"]))
    report = {"status": "passed", "primary_runs": len(primary), "budget_runs": len(budget),
              "active_entropy_ablation_runs": len(paired), "new_experiments": 0,
              "source": "retained unrounded per-run metrics; MedExpQA already filtered to121"}
    if a.write:
        for name, records in summaries.items():
            target = RESULTS / name
            text = csv_text(records)
            if target.exists() and target.read_text() != text.replace("\r\n", "\n"):
                raise ValueError(f"Refusing to overwrite different derived output {target}")
            target.write_text(text)
        (RESULTS / "aggregate_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
