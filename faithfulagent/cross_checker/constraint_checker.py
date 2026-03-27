"""Constraint verification helpers for cross-claim checks."""

import math

from faithfulagent.config import CrossClaimCheckConfig
from faithfulagent.schemas import Constraint, CrossClaimResult


def _severity_and_confidence(mode: str) -> tuple[str, str]:
    if mode == "strict":
        return "error", "high"
    if mode == "heuristic":
        return "warning", "medium"
    raise ValueError(f"Unknown mode: '{mode}'")


def _violation_category(constraint_type: str) -> str:
    if constraint_type == "comparison":
        return "comparison_violated"
    if constraint_type == "ranking":
        return "ranking_violated"
    if constraint_type == "sum":
        return "sum_mismatch"
    raise ValueError(f"Unknown constraint_type: '{constraint_type}'")


def _coercion_failure_result(
    constraint: Constraint,
    failing_claim_id: str,
    raw_value: float | None,
    mode: str,
    *,
    observed_values: dict[str, float | None] | None = None,
    involved_claim_ids: list[str] | None = None,
    expected_relationship: str = "",
) -> CrossClaimResult:
    severity, match_confidence = _severity_and_confidence(mode)

    return CrossClaimResult(
        constraint_id=constraint.constraint_id,
        constraint_type=constraint.constraint_type,
        status="error",
        violation_category=_violation_category(constraint.constraint_type),
        severity=severity,
        match_confidence=match_confidence,
        involved_claim_ids=involved_claim_ids or [failing_claim_id],
        observed_values=observed_values or {failing_claim_id: raw_value},
        expected_relationship=expected_relationship,
        diff={
            "coercion_error": True,
            "failed_claim_id": failing_claim_id,
            "raw_value": raw_value,
        },
    )


def check_comparison(
    constraint: Constraint, value_map: dict[str, float | None], mode: str
) -> CrossClaimResult:
    lhs_claim_id = constraint.lhs_claim_id
    rhs_claim_id = constraint.rhs_claim_id
    operator = constraint.operator
    assert lhs_claim_id is not None
    assert rhs_claim_id is not None
    assert operator is not None

    involved_claim_ids = [lhs_claim_id, rhs_claim_id]

    lhs_value = value_map.get(lhs_claim_id)
    rhs_value = value_map.get(rhs_claim_id)
    observed_values = {lhs_claim_id: lhs_value, rhs_claim_id: rhs_value}

    if lhs_value is None:
        return _coercion_failure_result(
            constraint,
            lhs_claim_id,
            lhs_value,
            mode,
            observed_values=observed_values,
            involved_claim_ids=involved_claim_ids,
            expected_relationship=f"{lhs_value} {operator} {rhs_value}",
        )

    if rhs_value is None:
        return _coercion_failure_result(
            constraint,
            rhs_claim_id,
            rhs_value,
            mode,
            observed_values=observed_values,
            involved_claim_ids=involved_claim_ids,
            expected_relationship=f"{lhs_value} {operator} {rhs_value}",
        )

    if operator == "<":
        held = lhs_value < rhs_value
    elif operator == ">":
        held = lhs_value > rhs_value
    elif operator == "<=":
        held = lhs_value <= rhs_value
    elif operator == ">=":
        held = lhs_value >= rhs_value
    else:
        held = math.isclose(lhs_value, rhs_value, rel_tol=0, abs_tol=1e-9)

    severity, match_confidence = _severity_and_confidence(mode)
    return CrossClaimResult(
        constraint_id=constraint.constraint_id,
        constraint_type=constraint.constraint_type,
        status="pass" if held else "error",
        violation_category=None if held else "comparison_violated",
        severity=severity,
        match_confidence=match_confidence,
        involved_claim_ids=involved_claim_ids,
        observed_values={lhs_claim_id: lhs_value, rhs_claim_id: rhs_value},
        expected_relationship=f"{lhs_value} {operator} {rhs_value}",
        diff={
            "lhs": lhs_value,
            "rhs": rhs_value,
            "operator": operator,
            "held": held,
        },
    )


def check_ranking(
    constraint: Constraint, value_map: dict[str, float | None], mode: str
) -> CrossClaimResult:
    ranked_claim_id = constraint.ranked_claim_id
    rank_type = constraint.rank_type
    candidate_claim_ids = constraint.candidate_claim_ids
    assert ranked_claim_id is not None
    assert rank_type is not None
    assert candidate_claim_ids is not None

    observed_values = {cid: value_map.get(cid) for cid in candidate_claim_ids}

    # Check for coercion failures
    for claim_id in candidate_claim_ids:
        if observed_values[claim_id] is None:
            ranked_value = observed_values.get(ranked_claim_id)
            return _coercion_failure_result(
                constraint,
                claim_id,
                None,
                mode,
                observed_values=observed_values,
                involved_claim_ids=list(candidate_claim_ids),
                expected_relationship=(
                    f"rank={rank_type}, ranked={ranked_value}, actual_{rank_type}=unknown"
                ),
            )

    candidate_values = [observed_values[cid] for cid in candidate_claim_ids]
    ranked_value = observed_values[ranked_claim_id]
    assert ranked_value is not None

    actual = min(candidate_values) if rank_type == "min" else max(candidate_values)
    held = math.isclose(ranked_value, actual, rel_tol=0, abs_tol=1e-9)

    severity, match_confidence = _severity_and_confidence(mode)

    diff = None
    if not held:
        actual_claim_id = next(
            claim_id
            for claim_id in candidate_claim_ids
            if math.isclose(observed_values[claim_id], actual, rel_tol=0, abs_tol=1e-9)
        )
        diff = {
            "ranked_value": ranked_value,
            "rank_type": rank_type,
            "actual_min_or_max": actual,
            "actual_min_or_max_claim_id": actual_claim_id,
        }

    return CrossClaimResult(
        constraint_id=constraint.constraint_id,
        constraint_type=constraint.constraint_type,
        status="pass" if held else "error",
        violation_category=None if held else "ranking_violated",
        severity=severity,
        match_confidence=match_confidence,
        involved_claim_ids=list(candidate_claim_ids),
        observed_values=observed_values,
        expected_relationship=f"rank={rank_type}, ranked={ranked_value}, actual_{rank_type}={actual}",
        diff=diff,
    )


def check_sum(
    constraint: Constraint,
    value_map: dict[str, float | None],
    config: CrossClaimCheckConfig,
    mode: str,
) -> CrossClaimResult:
    total_claim_id = constraint.total_claim_id
    component_claim_ids = constraint.component_claim_ids
    assert total_claim_id is not None
    assert component_claim_ids is not None

    involved_claim_ids = [total_claim_id] + component_claim_ids
    observed_values = {cid: value_map.get(cid) for cid in involved_claim_ids}

    total = observed_values[total_claim_id]
    if total is None:
        return _coercion_failure_result(
            constraint,
            total_claim_id,
            total,
            mode,
            observed_values=observed_values,
            involved_claim_ids=involved_claim_ids,
            expected_relationship=f"sum=unknown, total={total}",
        )

    for claim_id in component_claim_ids:
        if observed_values[claim_id] is None:
            return _coercion_failure_result(
                constraint,
                claim_id,
                None,
                mode,
                observed_values=observed_values,
                involved_claim_ids=involved_claim_ids,
                expected_relationship=f"sum=unknown, total={total}",
            )

    component_values = [observed_values[cid] for cid in component_claim_ids]

    component_sum = sum(component_values)
    delta = component_sum - total

    if total == 0:
        held = abs(delta) < config.sum_tolerance
        diff = {
            "total": total,
            "component_sum": component_sum,
            "delta": delta,
        }
    else:
        relative_delta = delta / abs(total)
        held = abs(relative_delta) < config.sum_tolerance
        diff = {
            "total": total,
            "component_sum": component_sum,
            "delta": delta,
            "relative_delta": relative_delta,
        }

    severity, match_confidence = _severity_and_confidence(mode)
    return CrossClaimResult(
        constraint_id=constraint.constraint_id,
        constraint_type=constraint.constraint_type,
        status="pass" if held else "error",
        violation_category=None if held else "sum_mismatch",
        severity=severity,
        match_confidence=match_confidence,
        involved_claim_ids=involved_claim_ids,
        observed_values=observed_values,
        expected_relationship=f"sum={component_sum}, total={total}",
        diff=None if held else diff,
    )
