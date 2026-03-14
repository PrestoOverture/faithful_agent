import pytest
from pydantic import ValidationError

import json
import uuid

from faithfulagent import Claim, MatchResult, Report, ReportSummary, SourceRef, TextSpan, ToolCall


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
