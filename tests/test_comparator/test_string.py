"""Tests for faithfulagent/comparator/string.py — Phase 8, Step 8.1."""

import pytest
from rapidfuzz import fuzz

from faithfulagent.comparator.string import check_string
from faithfulagent.config import StringCheckConfig


DEFAULT = StringCheckConfig()  # case_sensitive=False, fuzzy_threshold=0.85


# ---------------------------------------------------------------------------
# Exact match
# ---------------------------------------------------------------------------

def test_exact_match():
    r = check_string("Delta", "Delta", DEFAULT)
    assert r.status == "pass"
    assert r.mismatch_category is None
    assert r.diff["similarity"] == 100.0


def test_case_insensitive_match():
    r = check_string("delta", "Delta", DEFAULT)
    assert r.status == "pass"
    assert r.diff["similarity"] == 100.0


def test_case_insensitive_all_caps():
    r = check_string("DELTA", "Delta", DEFAULT)
    assert r.status == "pass"


def test_case_sensitive_mismatch():
    cfg = StringCheckConfig(case_sensitive=True, fuzzy_threshold=0.0)
    r = check_string("delta", "Delta", cfg)
    # "delta" != "Delta" strictly; fuzzy_threshold=0.0 so any score passes as warning
    assert r.status in ("warning", "error")


def test_case_sensitive_exact_match():
    cfg = StringCheckConfig(case_sensitive=True)
    r = check_string("Delta", "Delta", cfg)
    assert r.status == "pass"


# ---------------------------------------------------------------------------
# Fuzzy match (warning zone)
# ---------------------------------------------------------------------------

def test_fuzzy_close_match_is_warning():
    # "Delt" vs "Delta" scores ~89, above the 0.85 threshold
    r = check_string("Delt", "Delta", DEFAULT)
    assert r.status == "warning"
    assert r.mismatch_category is None


def test_fuzzy_threshold_respected():
    # Score just above threshold → warning
    cfg = StringCheckConfig(fuzzy_threshold=0.5)
    r = check_string("Dleta", "Delta", cfg)
    assert r.status == "warning"


def test_fuzzy_threshold_boundary():
    # Set threshold to 1.0 (100) — only perfect matches pass, everything else is error
    cfg = StringCheckConfig(case_sensitive=True, fuzzy_threshold=1.0)
    r = check_string("Dleta", "Delta", cfg)
    assert r.status == "error"


# ---------------------------------------------------------------------------
# Complete mismatch
# ---------------------------------------------------------------------------

def test_complete_mismatch():
    r = check_string("Lufthansa", "Delta", DEFAULT)
    assert r.status == "error"
    assert r.mismatch_category == "exact_mismatch"


def test_very_different_strings():
    r = check_string("ZZZZZZZZZ", "Delta", DEFAULT)
    assert r.status == "error"
    assert r.mismatch_category == "exact_mismatch"


# ---------------------------------------------------------------------------
# diff always populated
# ---------------------------------------------------------------------------

def test_diff_on_pass():
    r = check_string("Delta", "Delta", DEFAULT)
    assert r.diff is not None
    assert "expected" in r.diff
    assert "observed" in r.diff
    assert "similarity" in r.diff


def test_diff_on_warning():
    r = check_string("Delt", "Delta", DEFAULT)
    assert r.diff is not None
    assert r.diff["similarity"] > 0


def test_diff_on_error():
    r = check_string("Lufthansa", "Delta", DEFAULT)
    assert r.diff is not None
    assert "similarity" in r.diff


def test_diff_values_preserved():
    r = check_string("United", "Delta", DEFAULT)
    assert r.diff["observed"] == "United"
    assert r.diff["expected"] == "Delta"


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------

def test_int_coerced_to_string():
    r = check_string(553, 553, DEFAULT)
    assert r.status == "pass"


def test_mixed_types_coerced():
    r = check_string("553", 553, DEFAULT)
    assert r.status == "pass"


# ---------------------------------------------------------------------------
# Severity alignment
# ---------------------------------------------------------------------------

def test_pass_severity_is_info():
    r = check_string("Delta", "Delta", DEFAULT)
    assert r.severity == "info"


def test_warning_severity_is_warning():
    r = check_string("Delt", "Delta", DEFAULT)
    assert r.severity == "warning"


def test_error_severity_is_error():
    r = check_string("Lufthansa", "Delta", DEFAULT)
    assert r.severity == "error"


# ---------------------------------------------------------------------------
# Substring containment: expected is substring of observed (and vice versa)
# ---------------------------------------------------------------------------


def test_expected_substring_of_observed_is_pass():
    # "China Eastern" ⊂ "China Eastern Airlines"
    r = check_string("China Eastern Airlines", "China Eastern", DEFAULT)
    assert r.status == "pass"
    assert r.mismatch_category is None
    assert r.diff["note"] == "substring_match"


def test_observed_substring_of_expected_is_pass():
    # reversed direction
    r = check_string("China Eastern", "China Eastern Airlines", DEFAULT)
    assert r.status == "pass"
    assert r.mismatch_category is None
    assert r.diff["note"] == "substring_match"


def test_no_containment_does_not_trigger_substring_match():
    # "Delta" and "United Airlines" share no containment
    r = check_string("Delta", "United Airlines", DEFAULT)
    assert r.status in ("warning", "error")
    assert r.diff.get("note") != "substring_match"


def test_substring_match_severity_is_info():
    r = check_string("China Eastern Airlines", "China Eastern", DEFAULT)
    assert r.severity == "info"
