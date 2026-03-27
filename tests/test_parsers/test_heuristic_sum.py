import warnings
from unittest.mock import patch

import pytest

from faithfulagent import verify
from faithfulagent.parsers import parse
from faithfulagent.parsers.heuristic import _detect_sum_constraints, parse_heuristic
from faithfulagent.schemas import Claim, TextSpan, ToolCall


_TC_EMPTY: list[ToolCall] = []
_TC: list[ToolCall] = [ToolCall(call_id="c1", tool_name="t", response={"x": 1})]


def _claim(claim_id: str, claim_type: str, start: int, end: int, text: str, extracted_value: str | None = None) -> Claim:
    return Claim(
        claim_id=claim_id,
        claim_type=claim_type,
        text_span=TextSpan(start=start, end=end, text=text),
        extracted_value=extracted_value if extracted_value is not None else text,
        source_ref=None,
    )


def _regex_fallback_parse(text: str):
    with patch(
        "faithfulagent.parsers.heuristic._load_spacy_model",
        side_effect=ImportError("spaCy not installed"),
    ):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            return parse_heuristic(text, _TC_EMPTY)


def test_basic_total_detection():
    text = "flights $340, hotel $450, transfers $60. Total: $850"
    claims = [
        _claim("claim_0", "currency", 8, 12, "$340", "340"),
        _claim("claim_1", "currency", 20, 24, "$450", "450"),
        _claim("claim_2", "currency", 36, 39, "$60", "60"),
        _claim("claim_3", "currency", 48, 52, "$850", "850"),
    ]

    constraints = _detect_sum_constraints(claims, text)

    assert len(constraints) == 1
    assert constraints[0].constraint_type == "sum"
    assert constraints[0].total_claim_id == "claim_3"
    assert constraints[0].component_claim_ids == ["claim_0", "claim_1", "claim_2"]


def test_no_trigger_keyword():
    text = "flights $340, hotel $450, transfers $60, cost $850"
    claims = [
        _claim("claim_0", "currency", 8, 12, "$340", "340"),
        _claim("claim_1", "currency", 20, 24, "$450", "450"),
        _claim("claim_2", "currency", 36, 39, "$60", "60"),
        _claim("claim_3", "currency", 46, 50, "$850", "850"),
    ]

    assert _detect_sum_constraints(claims, text) == []


def test_fewer_than_2_components_skipped():
    text = "price $100. Total: $100"
    claims = [
        _claim("claim_0", "currency", 6, 10, "$100", "100"),
        _claim("claim_1", "currency", 19, 23, "$100", "100"),
    ]

    assert _detect_sum_constraints(claims, text) == []


@pytest.mark.parametrize(
    "text",
    [
        "Fees $200, tax $300. Grand Total: $500",
        "Fees $200, tax $300. Subtotal: $500",
        "Fees $200, tax $300. Sum: $500",
        "Fees $200, tax $300. Overall cost: $500",
        "Fees $200, tax $300. Combined: $500",
    ],
)
def test_keyword_variants_detect_sum_constraints(text):
    claims = [
        _claim("claim_0", "currency", 5, 9, "$200", "200"),
        _claim("claim_1", "currency", 15, 19, "$300", "300"),
        _claim("claim_2", "currency", len(text) - 4, len(text), "$500", "500"),
    ]

    constraints = _detect_sum_constraints(claims, text)

    assert len(constraints) == 1
    assert constraints[0].constraint_type == "sum"
    assert constraints[0].total_claim_id == "claim_2"


def test_case_insensitive():
    text_upper = "Fee $200, tax $300. TOTAL: $500"
    text_lower = "Fee $200, tax $300. total: $500"
    claims_upper = [
        _claim("claim_0", "currency", 4, 8, "$200", "200"),
        _claim("claim_1", "currency", 14, 18, "$300", "300"),
        _claim("claim_2", "currency", 27, 31, "$500", "500"),
    ]
    claims_lower = [
        _claim("claim_0", "currency", 4, 8, "$200", "200"),
        _claim("claim_1", "currency", 14, 18, "$300", "300"),
        _claim("claim_2", "currency", 27, 31, "$500", "500"),
    ]

    assert len(_detect_sum_constraints(claims_upper, text_upper)) == 1
    assert len(_detect_sum_constraints(claims_lower, text_lower)) == 1


def test_constraint_ids_sequential():
    text = "Subtotal: $100. Tax: $10. Total: $110"
    claims = [
        _claim("claim_0", "currency", 10, 14, "$100", "100"),
        _claim("claim_1", "currency", 21, 24, "$10", "10"),
        _claim("claim_2", "currency", 33, 37, "$110", "110"),
    ]

    constraints = _detect_sum_constraints(claims, text)

    assert [constraint.constraint_id for constraint in constraints] == [
        "heuristic_sum_0",
        "heuristic_sum_1",
    ]


def test_constraint_built_regardless_of_math():
    text = "flights $340, hotel $450. Total: $500"
    claims = [
        _claim("claim_0", "currency", 8, 12, "$340", "340"),
        _claim("claim_1", "currency", 20, 24, "$450", "450"),
        _claim("claim_2", "currency", 33, 37, "$500", "500"),
    ]

    constraints = _detect_sum_constraints(claims, text)

    assert len(constraints) == 1
    assert constraints[0].total_claim_id == "claim_2"


def test_subtotal_included_as_component():
    text = "Subtotal: $100. Tax: $10. Total: $110"
    claims = [
        _claim("claim_0", "currency", 10, 14, "$100", "100"),
        _claim("claim_1", "currency", 21, 24, "$10", "10"),
        _claim("claim_2", "currency", 33, 37, "$110", "110"),
    ]

    constraints = _detect_sum_constraints(claims, text)
    total_constraint = next(constraint for constraint in constraints if constraint.total_claim_id == "claim_2")

    assert total_constraint.component_claim_ids == ["claim_0", "claim_1"]


def test_string_and_date_claims_excluded_from_components():
    text = "Hotel Delta, Jan 15, fee $200, tax $300. Total: $500"
    claims = [
        _claim("claim_str", "string", 6, 11, "Delta"),
        _claim("claim_date", "date", 13, 19, "Jan 15"),
        _claim("claim_0", "currency", 25, 29, "$200", "200"),
        _claim("claim_1", "currency", 35, 39, "$300", "300"),
        _claim("claim_2", "currency", 48, 52, "$500", "500"),
    ]

    constraints = _detect_sum_constraints(claims, text)

    assert len(constraints) == 1
    assert constraints[0].component_claim_ids == ["claim_0", "claim_1"]


def test_value_before_keyword():
    text = "fee $200, tax $650, $850 total"
    claims = [
        _claim("claim_0", "currency", 4, 8, "$200", "200"),
        _claim("claim_1", "currency", 14, 18, "$650", "650"),
        _claim("claim_2", "currency", 20, 24, "$850", "850"),
    ]

    constraints = _detect_sum_constraints(claims, text)

    assert len(constraints) == 1
    assert constraints[0].total_claim_id == "claim_2"


def test_no_claim_match_for_orphan_total():
    text = "fees $100, tax $200. Total: $999"
    claims = [
        _claim("claim_0", "currency", 5, 9, "$100", "100"),
        _claim("claim_1", "currency", 15, 19, "$200", "200"),
    ]

    assert _detect_sum_constraints(claims, text) == []


def test_parse_heuristic_returns_tuple():
    result = parse_heuristic("flights $340, hotel $450", _TC)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], list)
    assert isinstance(result[1], list)


def test_parse_heuristic_with_total_pattern():
    claims, constraints = parse_heuristic("flights $340, hotel $450, transfers $60. Total: $850", _TC)
    assert len(claims) == 4
    assert len(constraints) == 1
    assert constraints[0].constraint_type == "sum"


def test_parse_heuristic_no_total_pattern():
    claims, constraints = parse_heuristic("flights $340, hotel $450", _TC)
    assert len(claims) == 2
    assert constraints == []


def test_parse_heuristic_regex_fallback_with_total():
    claims, constraints = _regex_fallback_parse("price $100, tax $20. Total: $120")
    assert len(claims) == 3
    assert len(constraints) == 1
    assert constraints[0].constraint_type == "sum"


def test_parse_heuristic_regex_fallback_no_total():
    claims, constraints = _regex_fallback_parse("price $100, tax $20")
    assert len(claims) == 2
    assert constraints == []


def test_parse_dispatcher_heuristic_passes_constraints():
    claims, constraints = parse("flights $340, hotel $450, transfers $60. Total: $850", _TC, mode="heuristic")
    assert len(claims) == 4
    assert len(constraints) == 1


def test_verify_heuristic_sum_violation():
    tool_calls = [
        {
            "call_id": "c1",
            "tool_name": "trip",
            "response": {"flights": 340, "hotel": 450, "transfers": 60, "total": 900},
        }
    ]
    report = verify(
        "flights $340, hotel $450, transfers $60. Total: $900",
        tool_calls,
        mode="heuristic",
    )
    assert report.summary.verdict == "warn"
    assert len(report.cross_claim_results) == 1
    assert report.cross_claim_results[0].violation_category == "sum_mismatch"
    assert report.cross_claim_results[0].severity == "warning"
    assert report.cross_claim_results[0].match_confidence == "medium"


def test_verify_heuristic_sum_pass():
    tool_calls = [
        {
            "call_id": "c1",
            "tool_name": "trip",
            "response": {"flights": 340, "hotel": 450, "transfers": 60, "total": 850},
        }
    ]
    report = verify(
        "flights $340, hotel $450, transfers $60. Total: $850",
        tool_calls,
        mode="heuristic",
    )
    assert len(report.cross_claim_results) == 1
    assert report.cross_claim_results[0].status == "pass"


def test_verify_heuristic_no_total_unchanged():
    tool_calls = [
        {
            "call_id": "c1",
            "tool_name": "trip",
            "response": {"flights": 340, "hotel": 450},
        }
    ]
    report = verify("flights $340, hotel $450", tool_calls, mode="heuristic")
    assert report.cross_claim_results == []
