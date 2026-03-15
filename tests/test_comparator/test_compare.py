"""Tests for faithfulagent/comparator/__init__.py — Phase 9, Step 9.1."""

import pytest

from faithfulagent.comparator import compare
from faithfulagent.config import VerifyConfig
from faithfulagent.schemas import Claim, SourceRef, TextSpan, ToolCall


CONFIG = VerifyConfig()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strict_claim(claim_id, call_id, path, extracted, claim_type="numeric"):
    return Claim(
        claim_id=claim_id,
        claim_type=claim_type,
        text_span=TextSpan(start=0, end=len(str(extracted)), text=str(extracted)),
        extracted_value=extracted,
        source_ref=SourceRef(call_id=call_id, path=path),
    )


def heuristic_claim(claim_id, extracted, claim_type="numeric"):
    return Claim(
        claim_id=claim_id,
        claim_type=claim_type,
        text_span=TextSpan(start=0, end=len(str(extracted)), text=str(extracted)),
        extracted_value=extracted,
        source_ref=None,
    )


TC = ToolCall(call_id="c1", tool_name="get_flight", response={"price": 553, "airline": "Delta"})


# ---------------------------------------------------------------------------
# Numeric claims
# ---------------------------------------------------------------------------

def test_numeric_pass():
    claim = strict_claim("cl0", "c1", "$.price", "553")
    results = compare([claim], [TC], CONFIG)
    assert len(results) == 1
    r = results[0]
    assert r.status == "pass"
    assert r.claim_id == "cl0"
    assert r.matched_call_id == "c1"
    assert r.matched_field == "$.price"
    assert r.expected_value == 553
    assert r.observed_value == "553"


def test_numeric_mismatch():
    claim = strict_claim("cl0", "c1", "$.price", "450")
    results = compare([claim], [TC], CONFIG)
    assert results[0].status == "error"
    assert results[0].mismatch_category == "exact_mismatch"


def test_numeric_rounding():
    claim = strict_claim("cl0", "c1", "$.price", "550")
    results = compare([claim], [TC], CONFIG)
    assert results[0].status == "warning"
    assert results[0].mismatch_category == "rounding"


# ---------------------------------------------------------------------------
# String claims
# ---------------------------------------------------------------------------

def test_string_pass():
    claim = strict_claim("cl0", "c1", "$.airline", "Delta", claim_type="string")
    results = compare([claim], [TC], CONFIG)
    assert results[0].status == "pass"
    assert results[0].matched_field == "$.airline"


def test_string_mismatch():
    claim = strict_claim("cl0", "c1", "$.airline", "United", claim_type="string")
    results = compare([claim], [TC], CONFIG)
    assert results[0].status == "error"
    assert results[0].mismatch_category == "exact_mismatch"


def test_date_falls_back_to_string():
    tc = ToolCall(call_id="c1", tool_name="t", response={"departs": "2024-03-15"})
    claim = strict_claim("cl0", "c1", "$.departs", "2024-03-15", claim_type="date")
    results = compare([claim], [tc], CONFIG)
    assert results[0].status == "pass"


# ---------------------------------------------------------------------------
# Unsupported claim (no match)
# ---------------------------------------------------------------------------

def test_no_match_returns_unsupported():
    claim = heuristic_claim("cl0", "ZZZZZ", claim_type="string")
    results = compare([claim], [TC], CONFIG)
    assert results[0].status == "warning"
    assert results[0].mismatch_category == "unsupported"
    assert results[0].match_confidence == "low"


# ---------------------------------------------------------------------------
# Currency claims
# ---------------------------------------------------------------------------

def test_currency_same_symbol_numeric_pass():
    tc = ToolCall(call_id="c1", tool_name="t", response={"price": 553})
    claim = strict_claim("cl0", "c1", "$.price", "¥553", claim_type="currency")
    results = compare([claim], [tc], CONFIG)
    # ¥ in observed but no symbol in expected (numeric 553) → no symbol conflict
    assert results[0].status == "pass"


def test_currency_symbol_mismatch_is_unit_conversion_error():
    tc = ToolCall(call_id="c1", tool_name="t", response={"price": "¥553"})
    claim = strict_claim("cl0", "c1", "$.price", "$553", claim_type="currency")
    results = compare([claim], [tc], CONFIG)
    assert results[0].status == "error"
    assert results[0].mismatch_category == "unit_conversion"


def test_currency_matching_symbols_uses_numeric_result():
    tc = ToolCall(call_id="c1", tool_name="t", response={"price": "¥553"})
    claim = strict_claim("cl0", "c1", "$.price", "¥553", claim_type="currency")
    results = compare([claim], [tc], CONFIG)
    assert results[0].status == "pass"


# ---------------------------------------------------------------------------
# match_confidence
# ---------------------------------------------------------------------------

def test_strict_claim_match_confidence_high():
    claim = strict_claim("cl0", "c1", "$.price", "553")
    results = compare([claim], [TC], CONFIG)
    assert results[0].match_confidence == "high"


def test_heuristic_claim_match_confidence_medium():
    tc = ToolCall(call_id="c1", tool_name="t", response={"price": 553})
    claim = heuristic_claim("cl0", "553")
    results = compare([claim], [tc], CONFIG)
    assert results[0].match_confidence == "medium"


# ---------------------------------------------------------------------------
# Mixed claims
# ---------------------------------------------------------------------------

def test_mixed_claims_three_results():
    tc = ToolCall(call_id="c1", tool_name="t", response={"price": 553, "airline": "Delta"})
    claims = [
        strict_claim("cl0", "c1", "$.price", "553"),          # pass
        strict_claim("cl1", "c1", "$.airline", "Delta", claim_type="string"),  # pass
        strict_claim("cl2", "c1", "$.price", "450"),          # error
    ]
    results = compare(claims, [tc], CONFIG)
    assert len(results) == 3
    statuses = [r.status for r in results]
    assert statuses.count("pass") == 2
    assert statuses.count("error") == 1


def test_empty_claims_returns_empty():
    assert compare([], [TC], CONFIG) == []


def test_claim_id_propagated():
    claim = strict_claim("my_claim_id", "c1", "$.price", "553")
    results = compare([claim], [TC], CONFIG)
    assert results[0].claim_id == "my_claim_id"
