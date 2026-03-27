"""Tests for faithfulagent/reporter/ — generate_report() and formatters."""

import json

import pytest

from faithfulagent.reporter import generate_report
from faithfulagent.reporter.formatters import to_json, to_summary
from faithfulagent.schemas import CrossClaimResult, MatchResult


def _mr(status):
    return MatchResult(claim_id="c1", status=status, severity="info" if status == "pass" else status)


def _ccr(status, severity="error", constraint_type="comparison"):
    return CrossClaimResult(
        constraint_id="cc1",
        constraint_type=constraint_type,
        status=status,
        violation_category="comparison_violated" if status == "error" else None,
        severity=severity,
        match_confidence="high" if severity == "error" else "medium",
        involved_claim_ids=["c1", "c2"],
        observed_values={"c1": 100.0, "c2": 200.0},
        expected_relationship="100.0 < 200.0",
    )


def test_to_json_is_valid_json():
    r = generate_report([_mr("pass")], "strict")
    assert json.loads(to_json(r)) is not None


def test_to_json_contains_required_keys():
    r = generate_report([_mr("pass")], "strict")
    parsed = json.loads(to_json(r))
    assert "verification_id" in parsed
    assert "claims" in parsed
    assert "summary" in parsed


def test_to_summary_pass():
    r = generate_report([_mr("pass"), _mr("pass")], "strict")
    s = to_summary(r)
    assert s.startswith("PASS")
    assert "2/2" in s


def test_to_summary_warn():
    r = generate_report([_mr("pass"), _mr("warning")], "strict")
    s = to_summary(r)
    assert s.startswith("WARN")
    assert "1 passed" in s
    assert "1 warning(s)" in s


def test_to_summary_fail():
    r = generate_report([_mr("error")], "strict")
    s = to_summary(r)
    assert s.startswith("FAIL")
    assert "1 error(s)" in s


# ── cross-claim: generate_report verdict and counters ────────────────────────


def test_no_cross_claim_results_default():
    report = generate_report([_mr("pass")], "strict")
    assert report.cross_claim_results == []
    assert report.summary.cross_claim_violations == 0


def test_cross_claim_violations_count():
    report = generate_report(
        [_mr("pass")],
        "strict",
        [_ccr("error"), _ccr("pass"), _ccr("error", severity="warning")],
    )
    assert report.summary.cross_claim_violations == 2


def test_match_result_counters_unchanged():
    report = generate_report(
        [_mr("pass"), _mr("error")],
        "strict",
        [_ccr("error")],
    )
    assert report.summary.passed == 1
    assert report.summary.errors == 1
    assert report.summary.warnings == 0


def test_verdict_fail_from_cross_claim_strict():
    report = generate_report([_mr("pass"), _mr("pass")], "strict", [_ccr("error", severity="error")])
    assert report.summary.verdict == "fail"


def test_verdict_warn_from_cross_claim_heuristic():
    report = generate_report([_mr("pass"), _mr("pass")], "strict", [_ccr("error", severity="warning")])
    assert report.summary.verdict == "warn"


def test_verdict_pass_no_violations():
    report = generate_report([_mr("pass"), _mr("pass")], "strict", [_ccr("pass")])
    assert report.summary.verdict == "pass"


def test_verdict_fail_match_error_overrides_cross_claim_warning():
    report = generate_report([_mr("error")], "strict", [_ccr("error", severity="warning")])
    assert report.summary.verdict == "fail"


def test_verdict_warn_match_warning_plus_no_cross_claim():
    report = generate_report([_mr("warning")], "strict")
    assert report.summary.verdict == "warn"


def test_cross_claim_results_in_report():
    cross_claim_results = [_ccr("error"), _ccr("pass")]
    report = generate_report([_mr("pass")], "strict", cross_claim_results)
    assert report.cross_claim_results == cross_claim_results


# ── cross-claim: to_summary formatting ───────────────────────────────────────


def test_summary_no_violations_matches_v0_1():
    report = generate_report([_mr("pass"), _mr("warning")], "strict")
    summary = to_summary(report)
    assert "cross-claim" not in summary


def test_summary_with_one_violation():
    report = generate_report([_mr("pass")], "strict", [_ccr("error")])
    summary = to_summary(report)
    assert summary.endswith(", 1 cross-claim violation(s)")


def test_summary_with_multiple_violations():
    report = generate_report([_mr("pass")], "strict", [_ccr("error"), _ccr("error"), _ccr("pass")])
    summary = to_summary(report)
    assert summary.endswith(", 2 cross-claim violation(s)")


def test_summary_pass_verdict_no_violations():
    report = generate_report([_mr("pass"), _mr("pass")], "strict")
    summary = to_summary(report)
    assert summary.startswith("PASS:")
    assert "cross-claim" not in summary
