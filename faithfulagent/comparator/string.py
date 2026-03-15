"""String comparison with case-insensitive and fuzzy matching."""

from typing import Any

from rapidfuzz import fuzz

from faithfulagent.config import StringCheckConfig
from faithfulagent.schemas import MatchResult


def check_string(observed: Any, expected: Any, config: StringCheckConfig) -> MatchResult:
    obs_str = str(observed)
    exp_str = str(expected)

    if config.case_sensitive:
        exact_match = obs_str == exp_str
    else:
        exact_match = obs_str.lower() == exp_str.lower()

    if exact_match:
        return MatchResult(
            claim_id="",
            status="pass",
            mismatch_category=None,
            severity="info",
            diff={"expected": expected, "observed": observed, "similarity": 100.0},
        )

    score = fuzz.ratio(obs_str, exp_str)
    threshold = config.fuzzy_threshold * 100

    if score >= threshold:
        return MatchResult(
            claim_id="",
            status="warning",
            mismatch_category=None,
            severity="warning",
            diff={"expected": expected, "observed": observed, "similarity": score},
        )

    return MatchResult(
        claim_id="",
        status="error",
        mismatch_category="exact_mismatch",
        severity="error",
        diff={"expected": expected, "observed": observed, "similarity": score},
    )
