"""Unit and currency conversion detection via static lookup table."""

from typing import Any

from faithfulagent.comparator.numeric import _to_float
from faithfulagent.config import UnitCheckConfig
from faithfulagent.schemas import MatchResult

CONVERSION_FACTORS: dict[str, float | None] = {
    "km_mi": 1.60934,
    "mi_km": 0.62137,
    "kg_lb": 0.45359,
    "lb_kg": 2.20462,
    "celsius_fahrenheit": None,
    "m_ft": 0.3048,
    "ft_m": 3.28084,
}


def check_unit_mismatch(
    observed: Any, expected: Any, config: UnitCheckConfig
) -> MatchResult | None:
    obs_f = _to_float(observed)
    exp_f = _to_float(expected)
    if obs_f is None or exp_f is None:
        return None
    if exp_f == 0:
        return None

    ratio = obs_f / exp_f
    for factor in CONVERSION_FACTORS.values():
        if factor is None:
            continue
        if abs(ratio - factor) / factor <= 0.02:
            delta = abs(obs_f - exp_f)
            if config.allow_conversion:
                return MatchResult(
                    claim_id="",
                    status="warning",
                    mismatch_category="unit_conversion",
                    severity="warning",
                    diff={"expected": expected, "observed": observed, "delta": delta},
                    recommended_action="Verify unit conversion is intentional",
                )
            return MatchResult(
                claim_id="",
                status="error",
                mismatch_category="unit_conversion",
                severity="error",
                diff={"expected": expected, "observed": observed, "delta": delta},
                recommended_action=None,
            )

    return None
