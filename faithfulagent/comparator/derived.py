"""Derived value verification: checks if a claim is a sum, average, min, or max of tool values."""

from typing import Any

from faithfulagent.comparator.numeric import _to_float
from faithfulagent.config import DerivedCheckConfig
from faithfulagent.schemas import MatchResult, ToolCall


def _collect_numeric_values(value: Any, acc: list[float]) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _collect_numeric_values(child, acc)
        return
    if isinstance(value, list):
        for child in value:
            _collect_numeric_values(child, acc)
        return
    coerced = _to_float(value)
    if coerced is not None:
        acc.append(coerced)


def check_derived(
    observed: Any, tool_calls: list[ToolCall], config: DerivedCheckConfig
) -> MatchResult | None:
    obs_f = _to_float(observed)
    if obs_f is None:
        return None

    for tool_call in tool_calls:
        nums: list[float] = []
        _collect_numeric_values(tool_call.response, nums)
        if len(nums) < 2:
            continue

        total = sum(nums)
        average = total / len(nums)
        minimum = min(nums)
        maximum = max(nums)

        derived_values = {
            "sum": total,
            "average": average,
            "min": minimum,
            "max": maximum,
        }

        for operation, derived in derived_values.items():
            denom = max(abs(derived), 1e-9)
            if abs(obs_f - derived) / denom <= config.tolerance:
                return MatchResult(
                    claim_id="",
                    status="pass",
                    mismatch_category="derived",
                    severity="info",
                    diff={
                        "expected": derived,
                        "observed": observed,
                        "operation": operation,
                        "call_id": tool_call.call_id,
                    },
                )

    return None
