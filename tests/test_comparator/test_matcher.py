"""Tests for faithfulagent/comparator/matcher.py — Phase 6, Step 6.1."""

import pytest

from faithfulagent.comparator.matcher import _flatten_response, match_claim
from faithfulagent.schemas import Claim, SourceRef, TextSpan, ToolCall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_strict_claim(call_id: str, path: str, extracted: str, claim_type: str = "numeric") -> Claim:
    return Claim(
        claim_id="claim_0",
        claim_type=claim_type,
        text_span=TextSpan(start=0, end=len(extracted), text=extracted),
        extracted_value=extracted,
        source_ref=SourceRef(call_id=call_id, path=path),
    )


def make_heuristic_claim(extracted: str, claim_type: str = "numeric") -> Claim:
    return Claim(
        claim_id="claim_0",
        claim_type=claim_type,
        text_span=TextSpan(start=0, end=len(extracted), text=extracted),
        extracted_value=extracted,
        source_ref=None,
    )


TC1 = ToolCall(call_id="c1", tool_name="get_flight", response={"price": 553, "airline": "Delta"})
TC2 = ToolCall(call_id="c2", tool_name="get_hotel", response={"price": 200, "hotel": "Marriott"})


# ---------------------------------------------------------------------------
# _flatten_response
# ---------------------------------------------------------------------------

def test_flatten_simple():
    flat = _flatten_response("c1", {"price": 553, "airline": "Delta"})
    paths = {p for _, p, _ in flat}
    values = {v for _, _, v in flat}
    assert "$.price" in paths
    assert "$.airline" in paths
    assert 553 in values
    assert "Delta" in values


def test_flatten_nested():
    flat = _flatten_response("c1", {"data": {"price": 100, "name": "JFK"}})
    paths = {p for _, p, _ in flat}
    assert "$.data.price" in paths
    assert "$.data.name" in paths


def test_flatten_no_dicts_as_leaf():
    flat = _flatten_response("c1", {"nested": {"a": 1}})
    values = [v for _, _, v in flat]
    assert not any(isinstance(v, dict) for v in values)


def test_flatten_call_id_preserved():
    flat = _flatten_response("myid", {"x": 1})
    assert all(cid == "myid" for cid, _, _ in flat)


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------

def test_strict_numeric_match():
    claim = make_strict_claim("c1", "$.price", "553")
    call_id, path, value = match_claim(claim, [TC1])
    assert call_id == "c1"
    assert path == "$.price"
    assert value == 553


def test_strict_string_match():
    claim = make_strict_claim("c1", "$.airline", "Delta", claim_type="string")
    call_id, path, value = match_claim(claim, [TC1])
    assert call_id == "c1"
    assert path == "$.airline"
    assert value == "Delta"


def test_strict_missing_field_returns_none_value():
    claim = make_strict_claim("c1", "$.nonexistent", "999")
    call_id, path, value = match_claim(claim, [TC1])
    assert call_id == "c1"
    assert path == "$.nonexistent"
    assert value is None


def test_strict_nested_path():
    tc = ToolCall(call_id="c1", tool_name="t", response={"data": {"name": "JFK"}})
    claim = make_strict_claim("c1", "$.data.name", "JFK", claim_type="string")
    call_id, path, value = match_claim(claim, [tc])
    assert value == "JFK"
    assert path == "$.data.name"


# ---------------------------------------------------------------------------
# Heuristic mode — numeric
# ---------------------------------------------------------------------------

def test_heuristic_numeric_exact_match():
    claim = make_heuristic_claim("553")
    call_id, path, value = match_claim(claim, [TC1])
    assert call_id == "c1"
    assert value == 553


def test_heuristic_numeric_closest_match():
    # TC1 has price=553, TC2 has price=200; claim=550 → closest is 553
    claim = make_heuristic_claim("550")
    call_id, path, value = match_claim(claim, [TC1, TC2])
    assert value == 553


def test_heuristic_numeric_recency_tiebreak():
    # Two tool calls both have value 100; later one (c2) should win
    tc_a = ToolCall(call_id="c_a", tool_name="t1", response={"x": 100})
    tc_b = ToolCall(call_id="c_b", tool_name="t2", response={"y": 100})
    claim = make_heuristic_claim("100")
    call_id, _, _ = match_claim(claim, [tc_a, tc_b])
    assert call_id == "c_b"


def test_heuristic_numeric_currency_symbols_stripped():
    tc = ToolCall(call_id="c1", tool_name="t", response={"price": 553})
    claim = make_heuristic_claim("¥553", claim_type="currency")
    # currency type uses string matching branch, not numeric
    # so test with numeric type and currency-prefixed string
    claim2 = make_heuristic_claim("$553", claim_type="numeric")
    _, _, value = match_claim(claim2, [tc])
    assert value == 553


def test_heuristic_numeric_non_numeric_returns_none():
    claim = make_heuristic_claim("abc")
    result = match_claim(claim, [TC1])
    assert result == (None, None, None)


def test_heuristic_numeric_no_numeric_values_returns_none():
    tc = ToolCall(call_id="c1", tool_name="t", response={"name": "Delta"})
    claim = make_heuristic_claim("553")
    assert match_claim(claim, [tc]) == (None, None, None)


# ---------------------------------------------------------------------------
# Heuristic mode — string
# ---------------------------------------------------------------------------

def test_heuristic_string_exact_match():
    claim = make_heuristic_claim("Delta", claim_type="string")
    call_id, path, value = match_claim(claim, [TC1])
    assert call_id == "c1"
    assert value == "Delta"


def test_heuristic_string_fuzzy_match():
    claim = make_heuristic_claim("Dleta", claim_type="string")  # typo
    call_id, path, value = match_claim(claim, [TC1])
    assert value == "Delta"


def test_heuristic_string_below_threshold_returns_none():
    tc = ToolCall(call_id="c1", tool_name="t", response={"name": "Lufthansa"})
    claim = make_heuristic_claim("ZZZZZ", claim_type="string")
    assert match_claim(claim, [tc]) == (None, None, None)


def test_heuristic_string_no_string_values_returns_none():
    tc = ToolCall(call_id="c1", tool_name="t", response={"price": 553})
    claim = make_heuristic_claim("Delta", claim_type="string")
    assert match_claim(claim, [tc]) == (None, None, None)


def test_heuristic_date_uses_string_branch():
    tc = ToolCall(call_id="c1", tool_name="t", response={"departure": "2024-03-15"})
    claim = make_heuristic_claim("2024-03-15", claim_type="date")
    call_id, path, value = match_claim(claim, [tc])
    assert value == "2024-03-15"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_tool_calls_returns_none():
    claim = make_heuristic_claim("553")
    assert match_claim(claim, []) == (None, None, None)
