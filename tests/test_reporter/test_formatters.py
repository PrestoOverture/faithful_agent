"""Tests for faithfulagent/reporter/formatters.py — Phase 16."""

import json

import pytest

from faithfulagent.reporter import generate_report
from faithfulagent.reporter.formatters import to_json, to_summary
from faithfulagent.schemas import MatchResult


def _mr(status):
    return MatchResult(claim_id="c1", status=status, severity="info" if status == "pass" else status)


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
