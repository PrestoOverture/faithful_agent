"""Tests for faithfulagent/parsers/__init__.py — Phase 5, Step 5.2."""

import pytest

from faithfulagent.parsers import parse
from faithfulagent.schemas import Claim, Constraint, ToolCall

TOOL_CALL = ToolCall(
    call_id="call_1",
    tool_name="get_flight",
    response={"price": 553},
)

STRICT_OUTPUT = {
    "text": "costs 553",
    "refs": [{"span": [6, 9], "call_id": "call_1", "path": "$.price"}],
}


def test_strict_mode_delegates_and_returns_claims():
    claims, constraints = parse(STRICT_OUTPUT, [TOOL_CALL], "strict")
    assert len(claims) == 1
    assert constraints == []
    assert isinstance(claims[0], Claim)


def test_strict_is_default_mode():
    claims, constraints = parse(STRICT_OUTPUT, [TOOL_CALL])
    assert len(claims) == 1
    assert constraints == []


def test_heuristic_mode_delegates():
    # heuristic always succeeds now — falls back to regex-only if spaCy is absent
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        claims, constraints = parse("price is 553", [TOOL_CALL], "heuristic")
    assert isinstance(claims, list)
    assert constraints == []


def test_unknown_mode_raises_value_error():
    with pytest.raises(ValueError):
        parse(STRICT_OUTPUT, [TOOL_CALL], "invalid")


def test_unknown_mode_error_message():
    with pytest.raises(ValueError, match="foo"):
        parse(STRICT_OUTPUT, [TOOL_CALL], "foo")


def test_strict_mode_returns_claims_and_constraints_tuple():
    output = {
        "text": "Flight A costs 553. Flight B costs 600.",
        "refs": [
            {"span": [15, 18], "call_id": "call_cmp", "path": "$.flight_a_price"},
            {"span": [35, 38], "call_id": "call_cmp", "path": "$.flight_b_price"},
        ],
        "constraints": [
            {
                "constraint_id": "cmp_1",
                "constraint_type": "comparison",
                "lhs_ref_index": 0,
                "operator": "<",
                "rhs_ref_index": 1,
            }
        ],
    }
    tool_call = ToolCall(
        call_id="call_cmp",
        tool_name="get_flights",
        response={"flight_a_price": 553, "flight_b_price": 600},
    )

    claims, constraints = parse(output, [tool_call], "strict")

    assert len(claims) == 2
    assert len(constraints) == 1
    assert isinstance(claims[0], Claim)
    assert isinstance(constraints[0], Constraint)
    assert constraints[0].constraint_id == "cmp_1"


def test_heuristic_mode_returns_empty_constraint_list():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        claims, constraints = parse("price is 553", [TOOL_CALL], "heuristic")

    assert isinstance(claims, list)
    assert constraints == []
