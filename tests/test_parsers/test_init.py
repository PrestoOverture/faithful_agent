"""Tests for faithfulagent/parsers/__init__.py — Phase 5, Step 5.2."""

import pytest

from faithfulagent.parsers import parse
from faithfulagent.schemas import Claim, ToolCall

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
    claims = parse(STRICT_OUTPUT, [TOOL_CALL], "strict")
    assert len(claims) == 1
    assert isinstance(claims[0], Claim)


def test_strict_is_default_mode():
    claims = parse(STRICT_OUTPUT, [TOOL_CALL])
    assert len(claims) == 1


def test_heuristic_mode_delegates():
    # heuristic is now implemented; spaCy may or may not be installed
    try:
        claims = parse("price is 553", [TOOL_CALL], "heuristic")
        assert isinstance(claims, list)
    except ImportError:
        pass  # spaCy not installed — acceptable in base env


def test_unknown_mode_raises_value_error():
    with pytest.raises(ValueError):
        parse(STRICT_OUTPUT, [TOOL_CALL], "invalid")


def test_unknown_mode_error_message():
    with pytest.raises(ValueError, match="foo"):
        parse(STRICT_OUTPUT, [TOOL_CALL], "foo")
