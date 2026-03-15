"""Tests for faithfulagent/comparator/unit.py — Phase 16 coverage."""

from faithfulagent.comparator.unit import check_unit_mismatch
from faithfulagent.config import UnitCheckConfig

ALLOW = UnitCheckConfig(allow_conversion=True)
DENY  = UnitCheckConfig(allow_conversion=False)


def test_non_numeric_returns_none():
    assert check_unit_mismatch("abc", 100, ALLOW) is None


def test_expected_zero_returns_none():
    assert check_unit_mismatch(62, 0, ALLOW) is None


def test_km_to_mi_warning():
    r = check_unit_mismatch(62.14, 100, ALLOW)
    assert r is not None and r.status == "warning" and r.mismatch_category == "unit_conversion"


def test_km_to_mi_error_when_disallowed():
    r = check_unit_mismatch(62.14, 100, DENY)
    assert r is not None and r.status == "error"


def test_no_match_returns_none():
    assert check_unit_mismatch(99, 100, ALLOW) is None
