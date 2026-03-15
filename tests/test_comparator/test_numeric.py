"""Tests for faithfulagent/comparator/numeric.py — Phase 7, Step 7.1."""

import pytest

from faithfulagent.comparator.numeric import _to_float, check_numeric
from faithfulagent.config import NumericCheckConfig


DEFAULT = NumericCheckConfig()  # tolerance=0.0, allow_rounding=True, max_rounding_deviation=10.0


# ---------------------------------------------------------------------------
# _to_float helper
# ---------------------------------------------------------------------------

def test_to_float_int():
    assert _to_float(553) == 553.0

def test_to_float_float():
    assert _to_float(4.5) == 4.5

def test_to_float_string_int():
    assert _to_float("553") == 553.0

def test_to_float_string_decimal():
    assert _to_float("4.5") == 4.5

def test_to_float_comma_separated():
    assert _to_float("1,234.56") == 1234.56

def test_to_float_currency_dollar():
    assert _to_float("$553") == 553.0

def test_to_float_currency_yen():
    assert _to_float("¥553") == 553.0

def test_to_float_currency_euro():
    assert _to_float("€553") == 553.0

def test_to_float_currency_pound():
    assert _to_float("£553") == 553.0

def test_to_float_non_numeric_returns_none():
    assert _to_float("abc") is None

def test_to_float_none_returns_none():
    assert _to_float(None) is None

def test_to_float_dict_returns_none():
    assert _to_float({}) is None


# ---------------------------------------------------------------------------
# Exact match
# ---------------------------------------------------------------------------

def test_exact_match_int():
    r = check_numeric(553, 553, DEFAULT)
    assert r.status == "pass"
    assert r.mismatch_category is None
    assert r.diff["delta"] == 0.0

def test_exact_match_string_numeric():
    r = check_numeric("553", 553, DEFAULT)
    assert r.status == "pass"

def test_exact_match_float():
    r = check_numeric(4.5, 4.5, DEFAULT)
    assert r.status == "pass"


# ---------------------------------------------------------------------------
# Within tolerance
# ---------------------------------------------------------------------------

def test_within_tolerance():
    cfg = NumericCheckConfig(tolerance=0.01, allow_rounding=False)
    r = check_numeric(550, 553, cfg)
    # rel diff = 3/553 ≈ 0.0054 < 0.01 → pass
    assert r.status == "pass"
    assert r.mismatch_category is None

def test_outside_tolerance_with_rounding_disabled():
    cfg = NumericCheckConfig(tolerance=0.0, allow_rounding=False)
    r = check_numeric(450, 553, cfg)
    assert r.status == "error"
    assert r.mismatch_category == "exact_mismatch"


# ---------------------------------------------------------------------------
# Rounding
# ---------------------------------------------------------------------------

def test_rounding_warning():
    cfg = NumericCheckConfig(tolerance=0.0, allow_rounding=True, max_rounding_deviation=10.0)
    r = check_numeric(550, 553, cfg)
    assert r.status == "warning"
    assert r.mismatch_category == "rounding"

def test_rounding_exact_boundary():
    cfg = NumericCheckConfig(tolerance=0.0, allow_rounding=True, max_rounding_deviation=3.0)
    r = check_numeric(550, 553, cfg)
    assert r.status == "warning"
    assert r.mismatch_category == "rounding"

def test_rounding_exceeds_max_deviation():
    cfg = NumericCheckConfig(tolerance=0.0, allow_rounding=True, max_rounding_deviation=2.0)
    r = check_numeric(550, 553, cfg)
    assert r.status == "error"
    assert r.mismatch_category == "exact_mismatch"

def test_rounding_disabled_gives_error():
    cfg = NumericCheckConfig(tolerance=0.0, allow_rounding=False, max_rounding_deviation=10.0)
    r = check_numeric(550, 553, cfg)
    assert r.status == "error"
    assert r.mismatch_category == "exact_mismatch"


# ---------------------------------------------------------------------------
# Large mismatch
# ---------------------------------------------------------------------------

def test_large_mismatch():
    r = check_numeric(450, 553, DEFAULT)
    assert r.status == "error"
    assert r.mismatch_category == "exact_mismatch"


# ---------------------------------------------------------------------------
# Non-numeric / unparseable
# ---------------------------------------------------------------------------

def test_non_numeric_observed():
    r = check_numeric("abc", 553, DEFAULT)
    assert r.status == "error"
    assert r.mismatch_category == "exact_mismatch"
    assert r.diff["delta"] is None

def test_non_numeric_expected():
    r = check_numeric(553, "abc", DEFAULT)
    assert r.status == "error"
    assert r.diff["delta"] is None


# ---------------------------------------------------------------------------
# diff always populated
# ---------------------------------------------------------------------------

def test_diff_always_populated_on_pass():
    r = check_numeric(553, 553, DEFAULT)
    assert r.diff is not None
    assert "expected" in r.diff
    assert "observed" in r.diff
    assert "delta" in r.diff

def test_diff_always_populated_on_warning():
    r = check_numeric(550, 553, DEFAULT)
    assert r.diff is not None
    assert r.diff["delta"] == pytest.approx(3.0)

def test_diff_always_populated_on_error():
    r = check_numeric(450, 553, DEFAULT)
    assert r.diff is not None
    assert r.diff["delta"] == pytest.approx(103.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_expected_zero_no_division_error():
    r = check_numeric(0, 0, DEFAULT)
    assert r.status == "pass"

def test_expected_zero_nonzero_observed():
    cfg = NumericCheckConfig(tolerance=0.0, allow_rounding=False)
    r = check_numeric(5, 0, cfg)
    assert r.status == "error"

def test_currency_string_coerced():
    r = check_numeric("$553", 553, DEFAULT)
    assert r.status == "pass"
