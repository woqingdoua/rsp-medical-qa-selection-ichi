"""Dependency-free derivation and validation of fixed K=8 Relative-RSP.

The source ``margin`` column is *negative* clean top-two probability margin.
``rsp_drop`` already averages non-flip positive margin drops over all K passes,
including a zero contribution from every flipped pass.  It must not be divided
by the number of non-flips a second time.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Integral, Real
from typing import Iterable


METHOD = "rsp_relative_m1_k8"
METHOD2 = "rsp_relative_m2_k8"
METHOD3 = "rsp_relative_m3_k8"
METHOD_RMS_BALANCED = "rsp_rms_balanced_lambda_k8"
MC_PASSES = 8
PREFILTER_MULTIPLIER = 4
SENTINEL = -1e30
_ROUNDING_ULPS = 32


def ordered_uid_sha256(uids: Iterable[str]) -> str:
    """Hash ordered UIDs using the existing newline-separated, no-tail format."""
    values = list(uids)
    if any(not isinstance(uid, str) or not uid or "\n" in uid or "\r" in uid for uid in values):
        raise ValueError("UIDs must be nonempty strings without line breaks")
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _summary(values: list[float]) -> dict:
    return {
        "min": min(values),
        "max": max(values),
        "mean": math.fsum(values) / len(values),
        "unique_values": len(set(values)),
        "nonzero_rows": sum(value != 0.0 for value in values),
    }


def derive_relative_rows(
    rows: list[dict],
    budget: int = 128,
    epsilon: float = 1e-8,
    method: str = METHOD,
    expected_candidate_indices=None,
    prefilter_name: str = "answer_entropy",
) -> tuple[list[dict], dict]:
    """Copy source rows, add F + D / max(-margin, epsilon), and audit selection.

    Candidates must be exactly the stable top ``min(4 * budget, N)`` entropy
    rows, with finite F and D.  All other rows must have null F and D and receive
    ``SENTINEL``.  An optional source ``index`` must equal its zero-based row
    position.  Ranking ties use this position, never UID or an unstable sort.

    F must lie on the K=8 grid.  Source metadata must independently establish
    the actual layer and pass count: grid membership alone cannot prove K=8.
    Signs are checked strictly, with only 32-ULP tolerance for nonzero upper
    bounds and the pass grid.  A zero upper bound requires D == 0 exactly.
    No source values or derived scores are silently clipped or sign-corrected.
    """
    if isinstance(budget, bool) or not isinstance(budget, Integral) or budget < 1:
        raise ValueError("budget must be a positive integer")
    budget = int(budget)
    epsilon = _finite_number(epsilon, "epsilon")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("method must be a nonempty string")
    if len(rows) < budget:
        raise ValueError(f"Only {len(rows)} pool rows for budget {budget}")

    required = ("uid", "entropy", "margin", "rsp_flip", "rsp_drop", "rsp_prefilter_eligible")
    entropy: list[float] = []
    clean_margins: list[float] = []
    uids: list[str] = []
    seen_uids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} must be a dictionary")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Row {index} missing required fields: {missing}")
        if method in row:
            raise ValueError(f"Row {index} already contains method {method!r}; refusing to overwrite")
        if "index" in row and (
            isinstance(row["index"], bool)
            or not isinstance(row["index"], Integral)
            or row["index"] != index
        ):
            raise ValueError(f"Row {index} has misaligned source index {row['index']!r}")
        uid = row["uid"]
        if not isinstance(uid, str) or not uid or "\n" in uid or "\r" in uid:
            raise ValueError(f"Row {index} UID must be a nonempty string without line breaks")
        if uid in seen_uids:
            raise ValueError(f"Duplicate UID {uid!r} at row {index}")
        seen_uids.add(uid)
        uids.append(uid)
        entropy.append(_finite_number(row["entropy"], f"Row {index} entropy"))
        source_margin = _finite_number(row["margin"], f"Row {index} margin")
        clean_margin = -source_margin
        if not 0.0 <= clean_margin <= 1.0:
            raise ValueError(
                f"Row {index} requires negative clean margin in [-1, 0]; "
                f"got margin={source_margin!r}"
            )
        # Normalize signed zero only in diagnostics, never in the source row.
        clean_margins.append(0.0 if clean_margin == 0.0 else clean_margin)
        if not isinstance(row["rsp_prefilter_eligible"], bool):
            raise ValueError(f"Row {index} rsp_prefilter_eligible must be a boolean")

    candidate_count = min(PREFILTER_MULTIPLIER * budget, len(rows))
    entropy_order = sorted(range(len(rows)), key=lambda index: (-entropy[index], index))
    if expected_candidate_indices is None:
        expected_order = entropy_order[:candidate_count]
    else:
        expected_order = [int(index) for index in expected_candidate_indices]
        if (len(set(expected_order)) != len(expected_order)
                or any(index < 0 or index >= len(rows) for index in expected_order)):
            raise ValueError("External prefilter candidates must be unique and in range")
        # A no-prefilter ablation is permitted only when every item is eligible.
        # Other external prefilters retain the established top-4B candidate size.
        required_count = len(rows) if prefilter_name == "none" else candidate_count
        if len(expected_order) != required_count:
            raise ValueError("External prefilter candidates have an unexpected size")
        candidate_count = required_count
    expected_candidates = set(expected_order)
    actual_candidates = {
        index for index, row in enumerate(rows) if row["rsp_prefilter_eligible"]
    }
    if actual_candidates != expected_candidates:
        missing = sorted(expected_candidates - actual_candidates)
        unexpected = sorted(actual_candidates - expected_candidates)
        label = "Entropy" if prefilter_name == "answer_entropy" else prefilter_name
        raise ValueError(
            f"{label} prefilter mismatch: expected exactly "
            f"{candidate_count} frozen candidate rows; "
            f"missing_indices={missing[:10]}, unexpected_indices={unexpected[:10]}"
        )

    eligible = sorted(actual_candidates)
    output = [dict(row, **{method: SENTINEL}) for row in rows]
    flips: list[float] = []
    drops: list[float] = []
    raw_margins: list[float] = []
    effective_margins: list[float] = []
    scores: list[float] = []
    bound_excesses: list[float] = []
    score_bound_excesses: list[float] = []
    for index, row in enumerate(rows):
        if index not in actual_candidates:
            if row["rsp_flip"] is not None or row["rsp_drop"] is not None:
                raise ValueError(f"Excluded row {index} must have null rsp_flip and rsp_drop")
            continue
        flip = _finite_number(row["rsp_flip"], f"Row {index} rsp_flip")
        drop = _finite_number(row["rsp_drop"], f"Row {index} rsp_drop")
        if not 0.0 <= flip <= 1.0:
            raise ValueError(f"Row {index} rsp_flip must lie in [0, 1]")
        scaled_flip = flip * MC_PASSES
        if abs(scaled_flip - round(scaled_flip)) > _ROUNDING_ULPS * math.ulp(max(1.0, scaled_flip)):
            raise ValueError(f"Row {index} rsp_flip is not on the K={MC_PASSES} grid")
        if drop < 0.0:
            raise ValueError(f"Row {index} rsp_drop must be nonnegative")
        raw_margin = clean_margins[index]
        bound = (1.0 - flip) * raw_margin
        tolerance = _ROUNDING_ULPS * math.ulp(bound) if bound > 0.0 else 0.0
        if drop > bound + tolerance:
            raise ValueError(
                f"Row {index} rsp_drop={drop!r} exceeds nonflip bound "
                f"(1-F)*m0={bound!r}; check margin sign and all-K averaging"
            )
        effective_margin = max(raw_margin, epsilon)
        score = flip + drop / effective_margin
        if not math.isfinite(score) or score > 1.0 + _ROUNDING_ULPS * math.ulp(1.0):
            raise ValueError(f"Row {index} relative score violates finite [0, 1] bound: {score!r}")
        output[index][method] = score
        flips.append(flip)
        drops.append(drop)
        raw_margins.append(raw_margin)
        effective_margins.append(effective_margin)
        scores.append(score)
        bound_excesses.append(max(0.0, drop - bound))
        score_bound_excesses.append(max(0.0, score - 1.0))

    selected = sorted(eligible, key=lambda index: (-output[index][method], index))[:budget]
    selected_uids = [uids[index] for index in selected]
    diagnostics = {
        "method": method,
        "formula": "rsp_flip + rsp_drop / max(-margin, epsilon)",
        "source_margin_convention": "negative_clean_top2_probability_margin",
        "drop_averaging": "mean_over_all_K_with_zero_on_flips",
        "mc_passes_grid_checked": MC_PASSES,
        "epsilon": epsilon,
        "sentinel": SENTINEL,
        "pool_rows": len(rows),
        "eligible_rows": len(eligible),
        "excluded_rows": len(rows) - len(eligible),
        "budget_items": budget,
        "prefilter_name": prefilter_name,
        "prefilter_multiplier": PREFILTER_MULTIPLIER,
        "prefilter_expected_rows": candidate_count,
        "prefilter_exact_match": True,
        "tie_break": "ascending_zero_based_source_row_index",
        "flip": _summary(flips),
        "drop": _summary(drops),
        "relative_score": _summary(scores),
        "raw_margin": {
            **_summary(raw_margins),
            "zero_rows": sum(value == 0.0 for value in raw_margins),
            "near_zero_rows": sum(value < epsilon for value in raw_margins),
            "near_zero_definition": "m0 < epsilon",
        },
        "effective_margin": _summary(effective_margins),
        "bound_check": {
            "drop_bound": "0 <= D <= (1-F)*m0",
            "score_bound": "0 <= relative_score <= 1",
            "rounding_tolerance_ulps": _ROUNDING_ULPS,
            "drop_max_rounding_excess": max(bound_excesses),
            "drop_rounding_excess_rows": sum(value > 0.0 for value in bound_excesses),
            "score_max_rounding_excess": max(score_bound_excesses),
            "score_rounding_excess_rows": sum(value > 0.0 for value in score_bound_excesses),
        },
        "pool_uid_sha256": ordered_uid_sha256(uids),
        "candidate_uid_sha256": ordered_uid_sha256([uids[index] for index in eligible]),
        "candidate_prefilter_order_uid_sha256": ordered_uid_sha256(
            [uids[index] for index in expected_order]),
        "selected_indices": selected,
        "selected_uids": selected_uids,
        "selected_uid_sha256": ordered_uid_sha256(selected_uids),
    }
    return output, diagnostics


def derive_rms_balanced_rows(
    rows: list[dict],
    budget: int = 128,
    epsilon: float = 1e-8,
    method: str = METHOD_RMS_BALANCED,
    expected_candidate_indices=None,
    prefilter_name: str = "answer_entropy",
) -> tuple[list[dict], dict]:
    """Balance Flip and Drop by their candidate-pool RMS scales.

    The frozen candidate pool is validated by :func:`derive_relative_rows`.
    We then compute ``lambda0 = RMS(F) / (RMS(D) + epsilon)`` over every
    eligible candidate and rank by ``F + lambda0 * D``.  This uses no labels,
    rationale text, or development-set performance.
    """
    validation_method = method + "__validation_only"
    _, base = derive_relative_rows(
        rows, budget=budget, epsilon=epsilon, method=validation_method,
        expected_candidate_indices=expected_candidate_indices,
        prefilter_name=prefilter_name,
    )
    eligible = [index for index, row in enumerate(rows) if row["rsp_prefilter_eligible"]]
    flips = [float(rows[index]["rsp_flip"]) for index in eligible]
    drops = [float(rows[index]["rsp_drop"]) for index in eligible]
    rms_flip = math.sqrt(math.fsum(value * value for value in flips) / len(flips))
    rms_drop = math.sqrt(math.fsum(value * value for value in drops) / len(drops))
    lambda0 = rms_flip / (rms_drop + epsilon)
    if not math.isfinite(lambda0) or lambda0 < 0.0:
        raise ValueError(f"Invalid RMS-balanced lambda: {lambda0!r}")
    output = [dict(row, **{method: SENTINEL}) for row in rows]
    scores = []
    for index in eligible:
        score = float(rows[index]["rsp_flip"]) + lambda0 * float(rows[index]["rsp_drop"])
        if not math.isfinite(score) or score < 0.0:
            raise ValueError(f"Row {index} has invalid RMS-balanced score: {score!r}")
        output[index][method] = score
        scores.append(score)
    selected = sorted(eligible, key=lambda index: (-output[index][method], index))[:budget]
    selected_uids = [rows[index]["uid"] for index in selected]
    diagnostics = dict(base)
    diagnostics.update({
        "method": method,
        "formula": "rsp_flip + lambda0 * rsp_drop",
        "lambda_rule": "lambda0 = RMS_candidate(rsp_flip) / (RMS_candidate(rsp_drop) + epsilon)",
        "lambda0": lambda0,
        "rms_flip": rms_flip,
        "rms_drop": rms_drop,
        "weighted_drop_rms": lambda0 * rms_drop,
        "rms_candidate_count": len(eligible),
        "score": _summary(scores),
        "selected_indices": selected,
        "selected_uids": selected_uids,
        "selected_uid_sha256": ordered_uid_sha256(selected_uids),
    })
    return output, diagnostics


def derive_relative2_rows(
    rows: list[dict], clean_probabilities, candidate_indices,
    perturbed_probabilities, budget: int = 128, epsilon: float = 1e-8,
    method: str = METHOD2, upper_bound: float = 2.0,
) -> tuple[list[dict], dict]:
    """Compute a bounded mean signed-margin degradation score.

    The signed noisy margin compares the clean top-1 class with its strongest
    noisy competitor. Full pass-level probabilities are required because the
    old aggregate Flip value does not retain flip depth.
    """
    import numpy as np

    epsilon = _finite_number(epsilon, "epsilon")
    upper_bound = _finite_number(upper_bound, "upper_bound")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if upper_bound < 1:
        raise ValueError("upper_bound must be at least one")
    if isinstance(budget, bool) or not isinstance(budget, Integral) or budget < 1:
        raise ValueError("budget must be a positive integer")
    budget = int(budget)
    if len(rows) < budget:
        raise ValueError("pool is smaller than budget")
    clean = np.asarray(clean_probabilities, dtype=np.float64)
    indices = np.asarray(candidate_indices, dtype=np.int64)
    noisy = np.asarray(perturbed_probabilities, dtype=np.float64)
    if clean.ndim != 2 or clean.shape[0] != len(rows) or clean.shape[1] < 2:
        raise ValueError("clean probabilities must have shape [pool,C], C>=2")
    candidate_count = min(PREFILTER_MULTIPLIER * budget, len(rows))
    if indices.shape != (candidate_count,) or len(set(indices.tolist())) != candidate_count:
        raise ValueError("candidate indices must be unique and exactly top-4B length")
    if noisy.shape != (MC_PASSES, candidate_count, clean.shape[1]):
        raise ValueError("perturbed probabilities must have shape [8,top-4B,C]")
    for name, array, axis in (("clean", clean, 1), ("perturbed", noisy, 2)):
        if not np.isfinite(array).all() or (array < 0).any() or (array > 1).any():
            raise ValueError(f"{name} probabilities must be finite and in [0,1]")
        if not np.allclose(array.sum(axis=axis), 1, atol=2e-6, rtol=0):
            raise ValueError(f"{name} probability rows must sum to one")
    entropy = [_finite_number(row.get("entropy"), f"row {i} entropy") for i, row in enumerate(rows)]
    expected = np.asarray(sorted(range(len(rows)), key=lambda i: (-entropy[i], i))[:candidate_count])
    if not np.array_equal(indices, expected):
        raise ValueError("candidate indices do not match stable top-4B entropy order")
    eligible = set(indices.tolist())
    if any(bool(row.get("rsp_prefilter_eligible")) != (i in eligible) for i, row in enumerate(rows)):
        raise ValueError("source eligibility flags disagree with candidate indices")
    if any(method in row for row in rows):
        raise ValueError(f"refusing to overwrite existing {method}")
    uids = [row.get("uid") for row in rows]
    if any(not isinstance(uid, str) or not uid for uid in uids) or len(set(uids)) != len(rows):
        raise ValueError("UIDs must be unique nonempty strings")

    clean_winner = clean[indices].argmax(axis=1)
    clean_other = clean[indices].copy()
    clean_other[np.arange(candidate_count), clean_winner] = -np.inf
    m0 = clean[indices, clean_winner] - clean_other.max(axis=1)
    noisy_clean_winner = noisy[:, np.arange(candidate_count), clean_winner]
    noisy_other = noisy.copy()
    noisy_other[:, np.arange(candidate_count), clean_winner] = -np.inf
    mk = noisy_clean_winner - noisy_other.max(axis=2)
    per_pass = np.clip(1.0 - mk / np.maximum(m0[None, :], epsilon), 0.0, upper_bound)
    aggregate = per_pass.mean(axis=0)
    output = [dict(row, **{method: SENTINEL}) for row in rows]
    for position, index in enumerate(indices.tolist()):
        output[index][method] = float(aggregate[position])
    selected = sorted(indices.tolist(), key=lambda i: (-float(output[i][method]), i))[:budget]
    position_by_index = {int(index): position for position, index in enumerate(indices)}
    selected_positions = np.asarray([position_by_index[i] for i in selected])

    def saturation(block):
        hit = block == upper_bound
        means = block.mean(axis=0)
        return {"per_pass_count": int(hit.sum()), "per_pass_total": int(hit.size),
                "per_pass_fraction": float(hit.mean()),
                "rows_with_any_count": int(hit.any(axis=0).sum()),
                "rows_with_any_fraction": float(hit.any(axis=0).mean()),
                "aggregate_score_equals_upper_count": int((means == upper_bound).sum()),
                "aggregate_score_equals_upper_fraction": float((means == upper_bound).mean())}

    selected_uids = [uids[i] for i in selected]
    diagnostics = {
        "method": method,
        "formula": f"mean_k clip(1-signed_margin_k/max(clean_margin,epsilon),0,{upper_bound:g})",
        "signed_margin_reference": "clean top-1 versus strongest noisy competitor",
        "epsilon": epsilon, "upper_bound": upper_bound,
        "mc_passes": MC_PASSES, "pool_rows": len(rows),
        "eligible_rows": candidate_count, "budget_items": budget,
        "score": _summary([float(v) for v in aggregate]),
        "clean_margin": {**_summary([float(v) for v in m0]),
                         "zero_rows": int((m0 == 0).sum()),
                         "near_zero_rows": int((m0 < epsilon).sum())},
        "candidate_saturation_at_upper_bound": saturation(per_pass),
        "selected_saturation_at_upper_bound": saturation(per_pass[:, selected_positions]),
        "selected_indices": selected, "selected_uids": selected_uids,
        "selected_uid_sha256": ordered_uid_sha256(selected_uids),
        "candidate_uid_sha256": ordered_uid_sha256(uids[i] for i in indices.tolist()),
    }
    if upper_bound == 2.0:  # Preserve the completed Relative2 artifact schema.
        for prefix in ("candidate", "selected"):
            values = dict(diagnostics[f"{prefix}_saturation_at_upper_bound"])
            values["aggregate_score_equals_2_count"] = values["aggregate_score_equals_upper_count"]
            values["aggregate_score_equals_2_fraction"] = values["aggregate_score_equals_upper_fraction"]
            diagnostics[f"{prefix}_saturation_at_upper_2"] = values
    return output, diagnostics


def derive_relative3_rows(rows, clean_probabilities, candidate_indices,
                          perturbed_probabilities, budget=128, epsilon=1e-8,
                          method=METHOD3):
    """Upper-3 post-hoc exploratory variant of the signed-margin score."""
    return derive_relative2_rows(rows, clean_probabilities, candidate_indices,
                                 perturbed_probabilities, budget, epsilon,
                                 method, upper_bound=3.0)
