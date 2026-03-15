"""Numeric value comparison with tolerance and rounding detection."""

import math
from typing import Any

from faithfulagent.config import NumericCheckConfig
from faithfulagent.schemas import MatchResult


def _to_float(value: Any) -> float | None:
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


def check_numeric(
    observed: Any, expected: Any, config: NumericCheckConfig
) -> MatchResult:
    obs_f = _to_float(observed)
    exp_f = _to_float(expected)

    if obs_f is None or exp_f is None:
        return MatchResult(
            claim_id="",
            status="error",
            mismatch_category="exact_mismatch",
            severity="error",
            diff={"expected": expected, "observed": observed, "delta": None},
        )

    if obs_f == exp_f:
        return MatchResult(
            claim_id="",
            status="pass",
            mismatch_category=None,
            severity="info",
            diff={"expected": expected, "observed": observed, "delta": 0.0},
        )

    delta = abs(obs_f - exp_f)
    rel = delta / abs(exp_f) if exp_f != 0 else delta

    if rel <= config.tolerance or math.isclose(rel, config.tolerance):
        return MatchResult(
            claim_id="",
            status="pass",
            mismatch_category=None,
            severity="info",
            diff={"expected": expected, "observed": observed, "delta": delta},
        )

    if config.allow_rounding and delta <= config.max_rounding_deviation:
        return MatchResult(
            claim_id="",
            status="warning",
            mismatch_category="rounding",
            severity="warning",
            diff={"expected": expected, "observed": observed, "delta": delta},
        )

    return MatchResult(
        claim_id="",
        status="error",
        mismatch_category="exact_mismatch",
        severity="error",
        diff={"expected": expected, "observed": observed, "delta": delta},
    )
