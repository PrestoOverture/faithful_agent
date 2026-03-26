"""Tests for faithfulagent/config.py — Phase 3, Step 3.1."""

import pytest
from pydantic import ValidationError

from faithfulagent.config import (
    CrossClaimCheckConfig,
    DerivedCheckConfig,
    NumericCheckConfig,
    StringCheckConfig,
    UnitCheckConfig,
    VerifyConfig,
)


# --- Default construction ---

def test_verify_config_defaults():
    cfg = VerifyConfig()
    assert cfg.numeric.tolerance == 0.0
    assert cfg.numeric.allow_rounding is True
    assert cfg.numeric.max_rounding_deviation == 10.0
    assert cfg.string.case_sensitive is False
    assert cfg.string.fuzzy_threshold == 0.85
    assert cfg.unit.allow_conversion is True
    assert cfg.derived.tolerance == 0.01


def test_numeric_check_config_defaults():
    cfg = NumericCheckConfig()
    assert cfg.tolerance == 0.0
    assert cfg.allow_rounding is True
    assert cfg.max_rounding_deviation == 10.0


def test_string_check_config_defaults():
    cfg = StringCheckConfig()
    assert cfg.case_sensitive is False
    assert cfg.fuzzy_threshold == 0.85


def test_unit_check_config_defaults():
    cfg = UnitCheckConfig()
    assert cfg.allow_conversion is True


def test_derived_check_config_defaults():
    cfg = DerivedCheckConfig()
    assert cfg.tolerance == 0.01


# --- from_dict: partial override ---

def test_from_dict_numeric_override():
    cfg = VerifyConfig.from_dict({"numeric": {"tolerance": 0.05}})
    assert cfg.numeric.tolerance == 0.05
    # all other numeric fields retain defaults
    assert cfg.numeric.allow_rounding is True
    assert cfg.numeric.max_rounding_deviation == 10.0
    # other sub-configs unaffected
    assert cfg.string.fuzzy_threshold == 0.85
    assert cfg.unit.allow_conversion is True
    assert cfg.derived.tolerance == 0.01


def test_from_dict_string_override():
    cfg = VerifyConfig.from_dict({"string": {"case_sensitive": True, "fuzzy_threshold": 0.9}})
    assert cfg.string.case_sensitive is True
    assert cfg.string.fuzzy_threshold == 0.9
    assert cfg.numeric.tolerance == 0.0


def test_from_dict_empty_dict():
    cfg = VerifyConfig.from_dict({})
    assert cfg.numeric.tolerance == 0.0
    assert cfg.string.fuzzy_threshold == 0.85


def test_from_dict_all_sub_configs():
    cfg = VerifyConfig.from_dict({
        "numeric": {"tolerance": 0.1, "allow_rounding": False},
        "string": {"fuzzy_threshold": 0.7},
        "unit": {"allow_conversion": False},
        "derived": {"tolerance": 0.05},
    })
    assert cfg.numeric.tolerance == 0.1
    assert cfg.numeric.allow_rounding is False
    assert cfg.string.fuzzy_threshold == 0.7
    assert cfg.unit.allow_conversion is False
    assert cfg.derived.tolerance == 0.05


# --- extra="forbid" on sub-configs ---

def test_numeric_unknown_key_raises():
    with pytest.raises(ValidationError):
        NumericCheckConfig(**{"bogus": 1})


def test_string_unknown_key_raises():
    with pytest.raises(ValidationError):
        StringCheckConfig(**{"bogus": 1})


def test_unit_unknown_key_raises():
    with pytest.raises(ValidationError):
        UnitCheckConfig(**{"bogus": 1})


def test_derived_unknown_key_raises():
    with pytest.raises(ValidationError):
        DerivedCheckConfig(**{"bogus": 1})


def test_verify_config_unknown_key_raises():
    with pytest.raises(ValidationError):
        VerifyConfig(**{"bogus": 1})


def test_from_dict_unknown_key_in_sub_config_raises():
    with pytest.raises(ValidationError):
        VerifyConfig.from_dict({"numeric": {"bogus": 1}})


def test_cross_claim_check_config_defaults():
    cfg = CrossClaimCheckConfig()
    assert cfg.enabled is True
    assert cfg.sum_tolerance == 0.01


def test_cross_claim_check_config_extra_fields_rejected():
    with pytest.raises(ValidationError):
        CrossClaimCheckConfig(enabled=True, sum_tolerance=0.01, bogus=True)


def test_verify_config_defaults_include_cross_claim_defaults():
    cfg = VerifyConfig()
    assert cfg.cross_claim.enabled is True
    assert cfg.cross_claim.sum_tolerance == 0.01


def test_verify_config_from_dict_empty_dict_defaults_cross_claim():
    cfg = VerifyConfig.from_dict({})
    assert cfg.cross_claim.enabled is True
    assert cfg.cross_claim.sum_tolerance == 0.01


def test_verify_config_from_dict_cross_claim_enabled_override():
    cfg = VerifyConfig.from_dict({"cross_claim": {"enabled": False}})
    assert cfg.cross_claim.enabled is False
    assert cfg.cross_claim.sum_tolerance == 0.01


def test_verify_config_from_dict_cross_claim_sum_tolerance_override():
    cfg = VerifyConfig.from_dict({"cross_claim": {"sum_tolerance": 0.05}})
    assert cfg.cross_claim.enabled is True
    assert cfg.cross_claim.sum_tolerance == 0.05


def test_verify_config_from_dict_existing_v01_overrides_still_work():
    cfg = VerifyConfig.from_dict(
        {
            "numeric": {"tolerance": 0.1, "allow_rounding": False},
            "string": {"case_sensitive": True, "fuzzy_threshold": 0.9},
            "unit": {"allow_conversion": False},
            "derived": {"tolerance": 0.05},
        }
    )
    assert cfg.numeric.tolerance == 0.1
    assert cfg.numeric.allow_rounding is False
    assert cfg.string.case_sensitive is True
    assert cfg.string.fuzzy_threshold == 0.9
    assert cfg.unit.allow_conversion is False
    assert cfg.derived.tolerance == 0.05
    assert cfg.cross_claim.enabled is True
    assert cfg.cross_claim.sum_tolerance == 0.01
