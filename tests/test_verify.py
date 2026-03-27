"""End-to-end integration tests for verify()."""

import inspect
import json

import pytest

from faithfulagent import verify

# ── shared fixtures ───────────────────────────────────────────────────────────

RAW_TC = [{"call_id": "c1", "tool_name": "get_flights", "response": {"price": 553, "airline": "Delta"}}]

STRICT_ALL_PASS = {
    "text": "The flight costs 553 on Delta.",
    "refs": [
        {"span": [17, 20], "call_id": "c1", "path": "$.price"},
        {"span": [24, 29], "call_id": "c1", "path": "$.airline"},
    ],
}

OAI_TC = [{"id": "c1", "function": {"name": "get_flights", "arguments": "{}"}, "response": {"price": 553, "airline": "Delta"}}]
LC_TC  = [{"tool": "get_flights", "tool_call_id": "c1", "tool_input": {}, "output": {"price": 553, "airline": "Delta"}}]


# ── strict mode ───────────────────────────────────────────────────────────────

def test_strict_all_pass(strict_basic):
    r = verify(strict_basic["agent_output"], strict_basic["tool_calls"], mode="strict")
    assert r.summary.verdict == "pass"
    assert r.summary.total_claims == 2
    assert r.summary.passed == 2


def test_strict_mismatch(strict_mismatch):
    r = verify(strict_mismatch["agent_output"], strict_mismatch["tool_calls"], mode="strict")
    assert r.summary.verdict == "fail"
    assert r.summary.errors >= 1


def test_strict_rounding(strict_rounding):
    r = verify(strict_rounding["agent_output"], strict_rounding["tool_calls"], mode="strict")
    assert r.summary.verdict == "warn"
    assert any(c.mismatch_category == "rounding" for c in r.claims)


def test_strict_mixed_five_claims():
    tc = [{"call_id": "c1", "tool_name": "t", "response": {"a": 100, "b": 200, "c": "Delta", "d": "JFK", "e": 42}}]
    # a: pass, b: rounding (195 vs 200), c: pass, d: mismatch, e: exact_mismatch
    out = {
        "text": "100 195 Delta LAX 99",
        "refs": [
            {"span": [0, 3],  "call_id": "c1", "path": "$.a"},
            {"span": [4, 7],  "call_id": "c1", "path": "$.b"},
            {"span": [8, 13], "call_id": "c1", "path": "$.c"},
            {"span": [14, 17],"call_id": "c1", "path": "$.d"},
            {"span": [18, 20],"call_id": "c1", "path": "$.e"},
        ],
    }
    r = verify(out, tc, mode="strict")
    assert r.summary.total_claims == 5
    assert r.summary.verdict == "fail"


def test_strict_numeric_mismatch_category():
    out = {"text": "costs 450", "refs": [{"span": [6, 9], "call_id": "c1", "path": "$.price"}]}
    r = verify(out, RAW_TC, mode="strict")
    assert r.claims[0].mismatch_category == "exact_mismatch"
    assert r.claims[0].status == "error"


def test_strict_string_mismatch():
    out = {"text": "airline United", "refs": [{"span": [8, 14], "call_id": "c1", "path": "$.airline"}]}
    r = verify(out, RAW_TC, mode="strict")
    assert r.summary.verdict == "fail"


def test_strict_custom_tolerance_pass():
    # 530 vs 553 — ~4.2% diff, fails at default, passes at 5%
    out = {"text": "costs 530", "refs": [{"span": [6, 9], "call_id": "c1", "path": "$.price"}]}
    r_default = verify(out, RAW_TC, mode="strict")
    r_custom  = verify(out, RAW_TC, mode="strict", checks={"numeric": {"tolerance": 0.05}})
    assert r_default.summary.verdict == "fail"
    assert r_custom.summary.verdict == "pass"


def test_strict_report_mode_field():
    r = verify(STRICT_ALL_PASS, RAW_TC, mode="strict")
    assert r.mode == "strict"


# ── source formats ────────────────────────────────────────────────────────────

def test_openai_source_format():
    r = verify(STRICT_ALL_PASS, OAI_TC, mode="strict", source_format="openai")
    assert r.summary.verdict == "pass"


def test_langchain_source_format():
    r = verify(STRICT_ALL_PASS, LC_TC, mode="strict", source_format="langchain")
    assert r.summary.verdict == "pass"


# ── unit conversion and derived ───────────────────────────────────────────────

def test_strict_unit_conversion_detected():
    tc = [{"call_id": "c1", "tool_name": "t", "response": {"distance_km": 100}}]
    out = {"text": "62.14 km", "refs": [{"span": [0, 5], "call_id": "c1", "path": "$.distance_km"}]}
    r = verify(out, tc, mode="strict")
    assert r.claims[0].mismatch_category == "unit_conversion"


def test_strict_derived_sum_detected():
    tc = [{"call_id": "c1", "tool_name": "t", "response": {"a": 10, "b": 20, "c": 30}}]
    out = {"text": "total 60", "refs": [{"span": [6, 8], "call_id": "c1", "path": "$.a"}]}
    # $.a = 10, but agent says 60 = sum; derived check should catch it
    r = verify(out, tc, mode="strict")
    assert r.claims[0].mismatch_category == "derived"
    assert r.claims[0].status == "pass"


# ── error handling ────────────────────────────────────────────────────────────

def test_invalid_source_format():
    with pytest.raises(ValueError):
        verify({}, [], source_format="xml")


def test_unknown_mode():
    with pytest.raises(ValueError):
        verify({}, [], mode="foo")


def test_heuristic_mode_not_implemented_without_spacy():
    # Without spaCy installed, heuristic raises ImportError; with it, returns claims.
    try:
        r = verify("price is 553", [{"call_id": "c1", "tool_name": "t", "response": {"price": 553}}], mode="heuristic")
        assert isinstance(r.claims, list)
    except ImportError:
        pass  # acceptable when spaCy not installed


# ── edge cases ────────────────────────────────────────────────────────────────

def test_empty_tool_calls():
    out = {"text": "nothing", "refs": []}
    r = verify(out, [], mode="strict")
    assert r.summary.total_claims == 0
    assert r.summary.verdict == "pass"


def test_empty_refs():
    r = verify({"text": "hello", "refs": []}, RAW_TC, mode="strict")
    assert r.summary.total_claims == 0
    assert r.summary.verdict == "pass"


def test_report_is_json_serializable():
    import json
    r = verify(STRICT_ALL_PASS, RAW_TC, mode="strict")
    json.dumps(r.model_dump(mode="json"))  # must not raise


def test_report_has_verification_id():
    import re
    r = verify(STRICT_ALL_PASS, RAW_TC, mode="strict")
    assert re.match(r"^[0-9a-f\-]{36}$", r.verification_id)


def test_two_reports_have_different_verification_ids():
    r1 = verify(STRICT_ALL_PASS, RAW_TC, mode="strict")
    r2 = verify(STRICT_ALL_PASS, RAW_TC, mode="strict")
    assert r1.verification_id != r2.verification_id


# ── cross-claim wiring ──────────────────────────────────────────────────────

CROSS_TC = [
    {
        "call_id": "c1",
        "tool_name": "get_flights",
        "response": {"price_a": 553, "price_b": 480, "airline": "Delta"},
    }
]

STRICT_NO_CONSTRAINTS = {
    "text": "Flight A costs 553. Flight B costs 480.",
    "refs": [
        {"span": [15, 18], "call_id": "c1", "path": "$.price_a"},
        {"span": [35, 38], "call_id": "c1", "path": "$.price_b"},
    ],
}

STRICT_COMPARISON_VIOLATION = {
    "text": "Flight A costs 553. Flight B costs 480.",
    "refs": [
        {"span": [15, 18], "call_id": "c1", "path": "$.price_a"},
        {"span": [35, 38], "call_id": "c1", "path": "$.price_b"},
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


def test_verify_strict_no_constraints_backward_compat():
    report = verify(STRICT_NO_CONSTRAINTS, CROSS_TC, mode="strict")
    assert report.cross_claim_results == []
    assert report.summary.cross_claim_violations == 0


def test_verify_strict_with_comparison_constraint():
    report = verify(STRICT_COMPARISON_VIOLATION, CROSS_TC, mode="strict")
    assert len(report.cross_claim_results) == 1
    assert report.cross_claim_results[0].status == "error"
    assert report.cross_claim_results[0].violation_category == "comparison_violated"
    assert report.cross_claim_results[0].severity == "error"
    assert report.cross_claim_results[0].match_confidence == "high"
    assert report.summary.cross_claim_violations == 1


def test_verify_strict_cross_claim_disabled():
    report = verify(
        STRICT_COMPARISON_VIOLATION,
        CROSS_TC,
        mode="strict",
        checks={"cross_claim": {"enabled": False}},
    )
    assert report.cross_claim_results == []


def test_verify_signature_unchanged():
    signature = inspect.signature(verify)
    assert list(signature.parameters) == [
        "agent_output",
        "tool_calls",
        "mode",
        "source_format",
        "checks",
    ]


def test_verify_report_json_serializable_with_cross_claims():
    report = verify(STRICT_COMPARISON_VIOLATION, CROSS_TC, mode="strict")
    parsed = json.loads(report.model_dump_json())
    assert "cross_claim_results" in parsed


def test_strict_comparison_violation(strict_comparison):
    report = verify(
        strict_comparison["agent_output"],
        strict_comparison["tool_calls"],
        mode="strict",
    )
    assert report.summary.verdict == "fail"
    assert len(report.cross_claim_results) == 1
    assert report.cross_claim_results[0].violation_category == "comparison_violated"
    assert report.cross_claim_results[0].severity == "error"
    assert report.cross_claim_results[0].match_confidence == "high"


def test_strict_ranking_violation(strict_ranking):
    report = verify(
        strict_ranking["agent_output"],
        strict_ranking["tool_calls"],
        mode="strict",
    )
    assert report.summary.verdict == "fail"
    assert report.cross_claim_results[0].violation_category == "ranking_violated"
    assert report.cross_claim_results[0].diff["actual_min_or_max"] == 150.0


def test_strict_sum_violation(strict_sum):
    report = verify(strict_sum["agent_output"], strict_sum["tool_calls"], mode="strict")
    assert report.summary.verdict == "fail"
    assert report.cross_claim_results[0].violation_category == "sum_mismatch"


def test_strict_sum_pass():
    tool_calls = [
        {"call_id": "call_flights", "tool_name": "get_flights", "response": {"amount": 340}},
        {"call_id": "call_hotel", "tool_name": "get_hotel", "response": {"amount": 450}},
        {"call_id": "call_transfers", "tool_name": "get_transfers", "response": {"amount": 60}},
        {"call_id": "call_total", "tool_name": "get_total", "response": {"amount": 850}},
    ]
    agent_output = {
        "text": "Flights 340, hotel 450, transfers 60. Total: 850.",
        "refs": [
            {"span": [8, 11], "call_id": "call_flights", "path": "$.amount"},
            {"span": [19, 22], "call_id": "call_hotel", "path": "$.amount"},
            {"span": [34, 36], "call_id": "call_transfers", "path": "$.amount"},
            {"span": [45, 48], "call_id": "call_total", "path": "$.amount"},
        ],
        "constraints": [
            {
                "constraint_id": "sum_pass_1",
                "constraint_type": "sum",
                "total_ref_index": 3,
                "component_ref_indices": [0, 1, 2],
            }
        ],
    }
    report = verify(agent_output, tool_calls, mode="strict")
    assert report.cross_claim_results[0].status == "pass"


def test_strict_no_constraints_backward_compat_fixture(strict_basic):
    report = verify(strict_basic["agent_output"], strict_basic["tool_calls"], mode="strict")
    assert report.cross_claim_results == []
    assert report.summary.cross_claim_violations == 0
    assert json.loads(report.model_dump_json()) is not None


def test_heuristic_sum_violation():
    import warnings

    tool_calls = [
        {
            "call_id": "c1",
            "tool_name": "get_trip",
            "response": {"flights": 340, "hotel": 450, "transfers": 60, "total": 900},
        }
    ]
    text = "flights $340, hotel $450, transfers $60. Total: $900"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = verify(text, tool_calls, mode="heuristic")
    assert report.summary.verdict == "warn"
    assert report.cross_claim_results[0].violation_category == "sum_mismatch"
    assert report.cross_claim_results[0].severity == "warning"
    assert report.cross_claim_results[0].match_confidence == "medium"


def test_heuristic_sum_pass():
    import warnings

    tool_calls = [
        {
            "call_id": "c1",
            "tool_name": "get_trip",
            "response": {"flights": 340, "hotel": 450, "transfers": 60, "total": 850},
        }
    ]
    text = "flights $340, hotel $450, transfers $60. Total: $850"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = verify(text, tool_calls, mode="heuristic")
    assert report.cross_claim_results[0].status == "pass"


def test_cross_claim_disabled_with_fixture(strict_comparison):
    report = verify(
        strict_comparison["agent_output"],
        strict_comparison["tool_calls"],
        mode="strict",
        checks={"cross_claim": {"enabled": False}},
    )
    assert report.cross_claim_results == []
    assert report.summary.cross_claim_violations == 0


def test_cross_claim_independent_of_match_result():
    tool_calls = [
        {
            "call_id": "call_prices",
            "tool_name": "get_prices",
            "response": {"price_a": 700, "price_b": 900},
        }
    ]
    agent_output = {
        "text": "Flight A costs 100. Flight B costs 200.",
        "refs": [
            {"span": [15, 18], "call_id": "call_prices", "path": "$.price_a"},
            {"span": [35, 38], "call_id": "call_prices", "path": "$.price_b"},
        ],
        "constraints": [
            {
                "constraint_id": "cmp_pass_1",
                "constraint_type": "comparison",
                "lhs_ref_index": 0,
                "operator": "<",
                "rhs_ref_index": 1,
            }
        ],
    }
    report = verify(agent_output, tool_calls, mode="strict")
    assert report.cross_claim_results[0].status == "pass"
    assert report.summary.errors > 0


def test_all_fixtures_json_serializable(strict_comparison, strict_ranking, strict_sum):
    for fixture in (strict_comparison, strict_ranking, strict_sum):
        report = verify(fixture["agent_output"], fixture["tool_calls"], mode="strict")
        parsed = json.loads(report.model_dump_json())
        assert "cross_claim_results" in parsed
