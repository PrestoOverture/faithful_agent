import pytest
from pydantic import ValidationError

import json
import uuid

from faithfulagent import Claim, MatchResult, Report, ReportSummary, SourceRef, TextSpan, ToolCall
from faithfulagent.schemas import Constraint, CrossClaimResult


def test_textspan_valid():
    span = TextSpan(start=0, end=5, text="hello")
    assert span.start == 0
    assert span.end == 5
    assert span.text == "hello"


def test_textspan_end_lt_start_raises():
    with pytest.raises(ValidationError):
        TextSpan(start=5, end=3, text="hi")


def test_textspan_equal_start_end_valid():
    span = TextSpan(start=3, end=3, text="")
    assert span.start == span.end


def test_sourceref_default_transform():
    ref = SourceRef(call_id="call_1", path="$.price")
    assert ref.transform == "none"


def test_sourceref_custom_transform():
    ref = SourceRef(call_id="call_1", path="$.price", transform="round")
    assert ref.transform == "round"


def test_textspan_round_trip():
    span = TextSpan(start=0, end=5, text="hello")
    assert TextSpan(**span.model_dump()) == span


def test_sourceref_round_trip():
    ref = SourceRef(call_id="call_1", path="$.price", transform="none")
    assert SourceRef(**ref.model_dump()) == ref


# ToolCall tests

def test_toolcall_required_fields_only():
    tc = ToolCall(call_id="call_1", tool_name="get_flights", response={"price": 553})
    assert tc.request == {}
    assert tc.timestamp is None


def test_toolcall_nested_response():
    tc = ToolCall(
        call_id="call_1",
        tool_name="get_flights",
        response={"data": {"price": 553, "airline": "Delta"}},
    )
    assert tc.response["data"]["price"] == 553


def test_toolcall_missing_response_raises():
    with pytest.raises(ValidationError):
        ToolCall(call_id="call_1", tool_name="get_flights")


# Claim tests

def test_claim_strict_mode():
    span = TextSpan(start=10, end=13, text="553")
    ref = SourceRef(call_id="call_1", path="$.price")
    claim = Claim(
        claim_id="claim_0",
        claim_type="numeric",
        text_span=span,
        extracted_value="553",
        source_ref=ref,
    )
    assert claim.claim_type == "numeric"
    assert claim.extracted_value == "553"
    assert claim.source_ref.path == "$.price"


def test_claim_heuristic_mode():
    span = TextSpan(start=0, end=5, text="Delta")
    claim = Claim(
        claim_id="claim_1",
        claim_type="string",
        text_span=span,
        extracted_value="Delta",
        source_ref=None,
    )
    assert claim.source_ref is None


def test_claim_invalid_type_raises():
    with pytest.raises(ValidationError):
        Claim(
            claim_id="claim_2",
            claim_type="boolean",
            text_span=TextSpan(start=0, end=4, text="true"),
            extracted_value="true",
        )


# MatchResult tests

def test_matchresult_passing():
    r = MatchResult(
        claim_id="claim_0",
        status="pass",
        severity="info",
        mismatch_category=None,
    )
    assert r.status == "pass"
    assert r.mismatch_category is None
    assert r.match_confidence == "high"


def test_matchresult_with_mismatch():
    r = MatchResult(
        claim_id="claim_0",
        matched_call_id="call_1",
        matched_field="$.price",
        expected_value=553,
        observed_value="450",
        status="error",
        mismatch_category="exact_mismatch",
        severity="error",
        diff={"expected": 553, "observed": 450, "delta": 103},
    )
    assert r.mismatch_category == "exact_mismatch"
    assert r.diff["delta"] == 103


def test_matchresult_invalid_status_raises():
    with pytest.raises(ValidationError):
        MatchResult(claim_id="claim_0", status="unknown", severity="info")


# ReportSummary + Report tests

def _make_summary(**kwargs):
    defaults = dict(total_claims=0, passed=0, warnings=0, errors=0, verdict="pass")
    return ReportSummary(**{**defaults, **kwargs})


def test_report_empty_claims():
    report = Report(mode="strict", summary=_make_summary(), claims=[])
    assert report.summary.verdict == "pass"
    assert report.claims == []


def test_report_autogenerates_id_and_timestamp():
    report = Report(mode="strict", summary=_make_summary(), claims=[])
    assert uuid.UUID(report.verification_id, version=4)
    assert report.timestamp is not None


def test_report_json_serializable():
    report = Report(mode="strict", summary=_make_summary(), claims=[])
    dumped = report.model_dump(mode="json")
    json.dumps(dumped)  # must not raise


def test_public_api_imports():
    from faithfulagent import Claim, MatchResult, Report, ReportSummary, SourceRef, TextSpan, ToolCall
    for cls in [Claim, MatchResult, Report, ReportSummary, SourceRef, TextSpan, ToolCall]:
        assert cls is not None


def _make_cross_claim_result(**kwargs):
    defaults = {
        "constraint_id": "constraint_1",
        "constraint_type": "comparison",
        "status": "pass",
        "involved_claim_ids": ["claim_1", "claim_2"],
        "observed_values": {"claim_1": 480.0, "claim_2": 553.0},
        "expected_relationship": "480.0 < 553.0",
    }
    return CrossClaimResult(**{**defaults, **kwargs})


def test_v01_model_fields_unchanged():
    assert set(TextSpan.model_fields) == {"start", "end", "text"}
    assert set(SourceRef.model_fields) == {"call_id", "path", "transform"}
    assert set(ToolCall.model_fields) == {
        "call_id",
        "tool_name",
        "request",
        "response",
        "timestamp",
    }
    assert set(Claim.model_fields) == {
        "claim_id",
        "claim_type",
        "text_span",
        "extracted_value",
        "source_ref",
    }
    assert set(MatchResult.model_fields) == {
        "claim_id",
        "matched_call_id",
        "matched_field",
        "expected_value",
        "observed_value",
        "status",
        "mismatch_category",
        "severity",
        "match_confidence",
        "diff",
        "recommended_action",
    }


def test_constraint_valid_comparison_instantiates():
    constraint = Constraint(
        constraint_id="constraint_1",
        constraint_type="comparison",
        lhs_claim_id="claim_1",
        operator="<",
        rhs_claim_id="claim_2",
    )
    assert constraint.constraint_type == "comparison"


def test_constraint_valid_ranking_instantiates():
    constraint = Constraint(
        constraint_id="constraint_1",
        constraint_type="ranking",
        ranked_claim_id="claim_1",
        rank_type="min",
        candidate_claim_ids=["claim_1", "claim_2", "claim_3"],
    )
    assert constraint.constraint_type == "ranking"


def test_constraint_valid_sum_instantiates():
    constraint = Constraint(
        constraint_id="constraint_1",
        constraint_type="sum",
        total_claim_id="claim_total",
        component_claim_ids=["claim_1", "claim_2"],
    )
    assert constraint.constraint_type == "sum"


@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        (
            {
                "constraint_id": "constraint_1",
                "constraint_type": "comparison",
                "operator": "<",
                "rhs_claim_id": "claim_2",
            },
            "lhs_claim_id",
        ),
        (
            {
                "constraint_id": "constraint_1",
                "constraint_type": "comparison",
                "lhs_claim_id": "claim_1",
                "rhs_claim_id": "claim_2",
            },
            "operator",
        ),
        (
            {
                "constraint_id": "constraint_1",
                "constraint_type": "comparison",
                "lhs_claim_id": "claim_1",
                "operator": "<",
            },
            "rhs_claim_id",
        ),
    ],
)
def test_constraint_comparison_missing_required_fields_raise(payload, expected_message):
    with pytest.raises(ValidationError, match=expected_message):
        Constraint(**payload)


@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        (
            {
                "constraint_id": "constraint_1",
                "constraint_type": "ranking",
                "rank_type": "min",
                "candidate_claim_ids": ["claim_1", "claim_2"],
            },
            "ranked_claim_id",
        ),
        (
            {
                "constraint_id": "constraint_1",
                "constraint_type": "ranking",
                "ranked_claim_id": "claim_1",
                "candidate_claim_ids": ["claim_1", "claim_2"],
            },
            "rank_type",
        ),
        (
            {
                "constraint_id": "constraint_1",
                "constraint_type": "ranking",
                "ranked_claim_id": "claim_1",
                "rank_type": "min",
            },
            "candidate_claim_ids",
        ),
        (
            {
                "constraint_id": "constraint_1",
                "constraint_type": "ranking",
                "ranked_claim_id": "claim_3",
                "rank_type": "min",
                "candidate_claim_ids": ["claim_1", "claim_2"],
            },
            "ranked_claim_id",
        ),
        (
            {
                "constraint_id": "constraint_1",
                "constraint_type": "ranking",
                "ranked_claim_id": "claim_1",
                "rank_type": "min",
                "candidate_claim_ids": ["claim_1"],
            },
            "at least 2",
        ),
    ],
)
def test_constraint_ranking_invalid_shapes_raise(payload, expected_message):
    with pytest.raises(ValidationError, match=expected_message):
        Constraint(**payload)


@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        (
            {
                "constraint_id": "constraint_1",
                "constraint_type": "sum",
                "component_claim_ids": ["claim_1", "claim_2"],
            },
            "total_claim_id",
        ),
        (
            {
                "constraint_id": "constraint_1",
                "constraint_type": "sum",
                "total_claim_id": "claim_total",
            },
            "component_claim_ids",
        ),
        (
            {
                "constraint_id": "constraint_1",
                "constraint_type": "sum",
                "total_claim_id": "claim_total",
                "component_claim_ids": ["claim_1"],
            },
            "at least 2",
        ),
    ],
)
def test_constraint_sum_invalid_shapes_raise(payload, expected_message):
    with pytest.raises(ValidationError, match=expected_message):
        Constraint(**payload)


def test_constraint_invalid_type_raises():
    with pytest.raises(ValidationError, match="constraint_type"):
        Constraint(constraint_id="constraint_1", constraint_type="unknown")


def test_constraint_model_dump_json_is_valid_json():
    constraint = Constraint(
        constraint_id="constraint_1",
        constraint_type="sum",
        total_claim_id="claim_total",
        component_claim_ids=["claim_1", "claim_2"],
    )
    assert json.loads(constraint.model_dump_json())["constraint_id"] == "constraint_1"


def test_cross_claim_result_passing_instantiates_with_none_violation_category():
    result = _make_cross_claim_result(status="pass", violation_category=None)
    assert result.status == "pass"
    assert result.violation_category is None


def test_cross_claim_result_failing_instantiates_with_violation_category():
    result = _make_cross_claim_result(
        status="error",
        violation_category="comparison_violated",
        diff={"lhs": 553.0, "rhs": 480.0, "operator": "<", "held": False},
    )
    assert result.status == "error"
    assert result.violation_category == "comparison_violated"


def test_cross_claim_result_defaults_are_error_and_high():
    result = _make_cross_claim_result()
    assert result.severity == "error"
    assert result.match_confidence == "high"


def test_cross_claim_result_model_dump_json_is_valid_json():
    result = _make_cross_claim_result(
        constraint_type="sum",
        involved_claim_ids=["claim_total", "claim_1", "claim_2"],
        observed_values={"claim_total": 900.0, "claim_1": 340.0, "claim_2": 560.0},
        expected_relationship="sum=900.0, total=900.0",
    )
    assert json.loads(result.model_dump_json())["constraint_type"] == "sum"


def test_report_summary_v01_constructor_defaults_cross_claim_violations_to_zero():
    summary = ReportSummary(total_claims=3, passed=3, warnings=0, errors=0, verdict="pass")
    assert summary.cross_claim_violations == 0


def test_report_summary_stores_explicit_cross_claim_violations():
    summary = ReportSummary(
        total_claims=3,
        passed=1,
        warnings=1,
        errors=1,
        cross_claim_violations=3,
        verdict="fail",
    )
    assert summary.cross_claim_violations == 3


def test_report_v01_constructor_defaults_cross_claim_results_and_summary_counter():
    report = Report(mode="strict", summary=_make_summary(), claims=[])
    assert report.cross_claim_results == []
    assert report.summary.cross_claim_violations == 0


def test_report_stores_populated_cross_claim_results():
    results = [
        _make_cross_claim_result(
            constraint_id="constraint_1",
            constraint_type="comparison",
            status="error",
            violation_category="comparison_violated",
        ),
        _make_cross_claim_result(
            constraint_id="constraint_2",
            constraint_type="sum",
            status="pass",
            involved_claim_ids=["claim_total", "claim_1", "claim_2"],
            observed_values={"claim_total": 850.0, "claim_1": 340.0, "claim_2": 510.0},
            expected_relationship="sum=850.0, total=850.0",
        ),
    ]
    report = Report(
        mode="strict",
        summary=_make_summary(cross_claim_violations=1, verdict="fail"),
        claims=[],
        cross_claim_results=results,
    )
    assert report.cross_claim_results == results
    assert report.summary.cross_claim_violations == 1


def test_report_model_dump_json_includes_cross_claim_results_and_cross_claim_violations():
    report = Report(
        mode="strict",
        summary=_make_summary(cross_claim_violations=2, verdict="fail"),
        claims=[],
        cross_claim_results=[
            _make_cross_claim_result(
                constraint_id="constraint_1",
                status="error",
                violation_category="comparison_violated",
            )
        ],
    )
    parsed = json.loads(report.model_dump_json())
    assert "cross_claim_results" in parsed
    assert parsed["summary"]["cross_claim_violations"] == 2
