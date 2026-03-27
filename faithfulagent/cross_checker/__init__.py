"""Cross-claim checking package."""

from faithfulagent.config import CrossClaimCheckConfig
from faithfulagent.cross_checker.constraint_checker import (
    check_comparison,
    check_ranking,
    check_sum,
)
from faithfulagent.schemas import (
    Claim,
    Constraint,
    CrossClaimResult,
    MatchResult,
)


def _to_float(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = (
            value.replace(",", "")
            .replace(" ", "")
            .replace("$", "")
            .replace("¥", "")
            .replace("€", "")
            .replace("£", "")
        )
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def check_cross_claims(
    claims: list[Claim],
    constraints: list[Constraint],
    match_results: list[MatchResult],
    config: CrossClaimCheckConfig,
    mode: str,
) -> list[CrossClaimResult]:
    if not config.enabled:
        return []

    _ = match_results

    value_map: dict[str, float | None] = {}
    for claim in claims:
        if claim.claim_type in ("numeric", "currency"):
            value_map[claim.claim_id] = _to_float(claim.extracted_value)

    results: list[CrossClaimResult] = []
    for constraint in constraints:
        if constraint.constraint_type == "comparison":
            results.append(check_comparison(constraint, value_map, mode))
        elif constraint.constraint_type == "ranking":
            results.append(check_ranking(constraint, value_map, mode))
        elif constraint.constraint_type == "sum":
            results.append(check_sum(constraint, value_map, config, mode))
        else:
            raise ValueError(f"Unknown constraint_type: '{constraint.constraint_type}'")

    return results


__all__ = ["check_cross_claims"]
