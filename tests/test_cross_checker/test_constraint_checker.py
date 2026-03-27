import importlib

import pytest

from faithfulagent.config import CrossClaimCheckConfig
from faithfulagent.cross_checker import check_cross_claims
from faithfulagent.cross_checker.constraint_checker import (
    check_comparison,
    check_ranking,
    check_sum,
)
from faithfulagent.schemas import Claim, Constraint, CrossClaimResult, MatchResult, TextSpan


def _comparison_constraint(operator: str = "<") -> Constraint:
    return Constraint(
        constraint_id="cmp_1",
        constraint_type="comparison",
        lhs_claim_id="claim_a",
        operator=operator,
        rhs_claim_id="claim_b",
    )


def _ranking_constraint(
    ranked_claim_id: str = "claim_a",
    rank_type: str = "min",
    candidate_claim_ids: list[str] | None = None,
) -> Constraint:
    return Constraint(
        constraint_id="rank_1",
        constraint_type="ranking",
        ranked_claim_id=ranked_claim_id,
        rank_type=rank_type,
        candidate_claim_ids=candidate_claim_ids or ["claim_a", "claim_b", "claim_c"],
    )


def _sum_constraint(
    total_claim_id: str = "claim_total",
    component_claim_ids: list[str] | None = None,
) -> Constraint:
    return Constraint(
        constraint_id="sum_1",
        constraint_type="sum",
        total_claim_id=total_claim_id,
        component_claim_ids=component_claim_ids or ["claim_flights", "claim_hotel", "claim_transfers"],
    )


def test_import_faithfulagent_cross_checker_succeeds():
    module = importlib.import_module("faithfulagent.cross_checker")
    assert module is not None


def test_check_comparison_553_less_than_480_fails():
    result = check_comparison(
        _comparison_constraint("<"),
        {"claim_a": 553.0, "claim_b": 480.0},
        "strict",
    )

    assert result.status == "error"
    assert result.violation_category == "comparison_violated"
    assert result.diff["held"] is False
    assert result.expected_relationship == "553.0 < 480.0"
    assert result.involved_claim_ids == ["claim_a", "claim_b"]
    assert result.severity == "error"
    assert result.match_confidence == "high"


def test_check_comparison_480_less_than_553_passes():
    result = check_comparison(
        _comparison_constraint("<"),
        {"claim_a": 480.0, "claim_b": 553.0},
        "strict",
    )

    assert result.status == "pass"
    assert result.violation_category is None
    assert result.diff["held"] is True
    assert result.expected_relationship == "480.0 < 553.0"
    assert result.involved_claim_ids == ["claim_a", "claim_b"]


@pytest.mark.parametrize(
    ("lhs_value", "operator", "rhs_value", "expected_status"),
    [
        (1.0, "<", 2.0, "pass"),
        (2.0, "<", 1.0, "error"),
        (2.0, ">", 1.0, "pass"),
        (1.0, ">", 2.0, "error"),
        (2.0, "<=", 2.0, "pass"),
        (3.0, "<=", 2.0, "error"),
        (2.0, ">=", 2.0, "pass"),
        (1.0, ">=", 2.0, "error"),
    ],
)
def test_check_comparison_each_operator_works(lhs_value, operator, rhs_value, expected_status):
    result = check_comparison(
        _comparison_constraint(operator),
        {"claim_a": lhs_value, "claim_b": rhs_value},
        "strict",
    )

    assert result.status == expected_status
    assert result.diff["held"] is (expected_status == "pass")


def test_check_comparison_equality_uses_isclose_tolerance():
    pass_result = check_comparison(
        _comparison_constraint("=="),
        {"claim_a": 200.0, "claim_b": 200.0 + 5e-10},
        "strict",
    )
    fail_result = check_comparison(
        _comparison_constraint("=="),
        {"claim_a": 200.0, "claim_b": 201.0},
        "strict",
    )

    assert pass_result.status == "pass"
    assert pass_result.diff["held"] is True
    assert fail_result.status == "error"
    assert fail_result.diff["held"] is False


def test_check_comparison_heuristic_mode_sets_warning_and_medium():
    result = check_comparison(
        _comparison_constraint("<"),
        {"claim_a": 480.0, "claim_b": 553.0},
        "heuristic",
    )

    assert result.severity == "warning"
    assert result.match_confidence == "medium"


def test_check_comparison_coercion_failure_includes_other_observed_values():
    result = check_comparison(
        _comparison_constraint("<"),
        {"claim_a": None, "claim_b": 480.0},
        "strict",
    )

    assert result.status == "error"
    assert result.diff["coercion_error"] is True
    assert result.diff["failed_claim_id"] == "claim_a"
    assert result.observed_values == {"claim_a": None, "claim_b": 480.0}
    assert result.severity == "error"
    assert result.match_confidence == "high"


def test_check_ranking_min_failure_reports_actual_min_and_claim_id():
    result = check_ranking(
        _ranking_constraint(ranked_claim_id="claim_a", rank_type="min"),
        {"claim_a": 200.0, "claim_b": 150.0, "claim_c": 220.0},
        "strict",
    )

    assert result.status == "error"
    assert result.violation_category == "ranking_violated"
    assert result.diff["actual_min_or_max"] == 150.0
    assert result.diff["actual_min_or_max_claim_id"] == "claim_b"
    assert result.expected_relationship == "rank=min, ranked=200.0, actual_min=150.0"
    assert result.involved_claim_ids == ["claim_a", "claim_b", "claim_c"]


def test_check_ranking_correct_max_passes():
    result = check_ranking(
        _ranking_constraint(ranked_claim_id="claim_c", rank_type="max"),
        {"claim_a": 200.0, "claim_b": 150.0, "claim_c": 220.0},
        "strict",
    )

    assert result.status == "pass"
    assert result.violation_category is None
    assert result.expected_relationship == "rank=max, ranked=220.0, actual_max=220.0"


def test_check_ranking_tie_for_min_passes():
    result = check_ranking(
        _ranking_constraint(ranked_claim_id="claim_a", rank_type="min"),
        {"claim_a": 150.0, "claim_b": 150.0, "claim_c": 220.0},
        "strict",
    )

    assert result.status == "pass"
    assert result.violation_category is None


@pytest.mark.parametrize("rank_type", ["min", "max"])
def test_check_ranking_all_candidates_equal_passes(rank_type):
    result = check_ranking(
        _ranking_constraint(ranked_claim_id="claim_b", rank_type=rank_type),
        {"claim_a": 200.0, "claim_b": 200.0, "claim_c": 200.0},
        "strict",
    )

    assert result.status == "pass"
    assert result.violation_category is None


def test_check_ranking_uses_isclose_tolerance():
    result = check_ranking(
        _ranking_constraint(ranked_claim_id="claim_a", rank_type="min"),
        {"claim_a": 200.0 + 5e-10, "claim_b": 200.0, "claim_c": 220.0},
        "strict",
    )

    assert result.status == "pass"
    assert result.violation_category is None


def test_check_ranking_heuristic_mode_sets_warning_and_medium():
    result = check_ranking(
        _ranking_constraint(ranked_claim_id="claim_c", rank_type="max"),
        {"claim_a": 200.0, "claim_b": 150.0, "claim_c": 220.0},
        "heuristic",
    )

    assert result.severity == "warning"
    assert result.match_confidence == "medium"


def test_check_ranking_coercion_failure_includes_other_observed_values():
    result = check_ranking(
        _ranking_constraint(ranked_claim_id="claim_a", rank_type="min"),
        {"claim_a": 200.0, "claim_b": None, "claim_c": 220.0},
        "strict",
    )

    assert result.status == "error"
    assert result.diff["coercion_error"] is True
    assert result.diff["failed_claim_id"] == "claim_b"
    assert result.observed_values == {"claim_a": 200.0, "claim_b": None, "claim_c": 220.0}
    assert result.severity == "error"
    assert result.match_confidence == "high"


def test_check_sum_mismatch_reports_delta_and_relative_delta():
    result = check_sum(
        _sum_constraint(),
        {
            "claim_total": 900.0,
            "claim_flights": 340.0,
            "claim_hotel": 450.0,
            "claim_transfers": 60.0,
        },
        CrossClaimCheckConfig(sum_tolerance=0.01),
        "strict",
    )

    assert result.status == "error"
    assert result.violation_category == "sum_mismatch"
    assert result.diff["delta"] == -50.0
    assert result.diff["relative_delta"] == pytest.approx(-50.0 / 900.0)
    assert result.expected_relationship == "sum=850.0, total=900.0"
    assert result.involved_claim_ids == [
        "claim_total",
        "claim_flights",
        "claim_hotel",
        "claim_transfers",
    ]


def test_check_sum_exact_total_passes():
    result = check_sum(
        _sum_constraint(),
        {
            "claim_total": 850.0,
            "claim_flights": 340.0,
            "claim_hotel": 450.0,
            "claim_transfers": 60.0,
        },
        CrossClaimCheckConfig(sum_tolerance=0.01),
        "strict",
    )

    assert result.status == "pass"
    assert result.violation_category is None
    assert result.expected_relationship == "sum=850.0, total=850.0"


def test_check_sum_relative_tolerance_passes():
    result = check_sum(
        _sum_constraint(),
        {
            "claim_total": 850.0,
            "claim_flights": 340.0,
            "claim_hotel": 450.0,
            "claim_transfers": 62.0,
        },
        CrossClaimCheckConfig(sum_tolerance=0.01),
        "strict",
    )

    assert result.status == "pass"
    assert result.violation_category is None


def test_check_sum_zero_total_with_non_zero_component_sum_fails_without_relative_delta():
    result = check_sum(
        _sum_constraint(total_claim_id="claim_total", component_claim_ids=["claim_a", "claim_b"]),
        {"claim_total": 0.0, "claim_a": 1.0, "claim_b": -0.5},
        CrossClaimCheckConfig(sum_tolerance=0.01),
        "strict",
    )

    assert result.status == "error"
    assert result.violation_category == "sum_mismatch"
    assert "relative_delta" not in result.diff


def test_check_sum_zero_total_with_zero_component_sum_passes():
    result = check_sum(
        _sum_constraint(total_claim_id="claim_total", component_claim_ids=["claim_a", "claim_b"]),
        {"claim_total": 0.0, "claim_a": 1.0, "claim_b": -1.0},
        CrossClaimCheckConfig(sum_tolerance=0.01),
        "strict",
    )

    assert result.status == "pass"
    assert result.violation_category is None


def test_check_sum_negative_total_is_handled_correctly():
    result = check_sum(
        _sum_constraint(total_claim_id="claim_total", component_claim_ids=["claim_a", "claim_b"]),
        {"claim_total": -150.0, "claim_a": -100.0, "claim_b": -50.0},
        CrossClaimCheckConfig(sum_tolerance=0.01),
        "strict",
    )

    assert result.status == "pass"
    assert result.violation_category is None
    assert result.expected_relationship == "sum=-150.0, total=-150.0"


def test_check_sum_heuristic_mode_sets_warning_and_medium():
    result = check_sum(
        _sum_constraint(),
        {
            "claim_total": 850.0,
            "claim_flights": 340.0,
            "claim_hotel": 450.0,
            "claim_transfers": 60.0,
        },
        CrossClaimCheckConfig(sum_tolerance=0.01),
        "heuristic",
    )

    assert result.severity == "warning"
    assert result.match_confidence == "medium"


def test_check_sum_coercion_failure_includes_other_observed_values():
    result = check_sum(
        _sum_constraint(),
        {
            "claim_total": 850.0,
            "claim_flights": 340.0,
            "claim_hotel": None,
            "claim_transfers": 60.0,
        },
        CrossClaimCheckConfig(sum_tolerance=0.01),
        "strict",
    )

    assert result.status == "error"
    assert result.diff["coercion_error"] is True
    assert result.diff["failed_claim_id"] == "claim_hotel"
    assert result.observed_values == {
        "claim_total": 850.0,
        "claim_flights": 340.0,
        "claim_hotel": None,
        "claim_transfers": 60.0,
    }
    assert result.severity == "error"
    assert result.match_confidence == "high"


def _claim(claim_id: str, claim_type: str, extracted_value: str) -> Claim:
    return Claim(
        claim_id=claim_id,
        claim_type=claim_type,
        text_span=TextSpan(start=0, end=len(extracted_value), text=extracted_value),
        extracted_value=extracted_value,
        source_ref=None,
    )


def _cross_claim_result(constraint_id: str, constraint_type: str, status: str = "pass") -> CrossClaimResult:
    return CrossClaimResult(
        constraint_id=constraint_id,
        constraint_type=constraint_type,
        status=status,
        violation_category=None if status == "pass" else "comparison_violated",
        involved_claim_ids=[],
        observed_values={},
        expected_relationship="",
    )


def _match_result() -> MatchResult:
    return MatchResult(claim_id="claim_x", status="pass", severity="info")


def test_import_check_cross_claims_from_faithfulagent_cross_checker_succeeds():
    assert callable(check_cross_claims)


def test_check_cross_claims_empty_constraints_returns_empty_list():
    claims = [_claim("claim_a", "numeric", "123")]

    results = check_cross_claims(
        claims,
        [],
        [],
        CrossClaimCheckConfig(enabled=True, sum_tolerance=0.01),
        "strict",
    )

    assert results == []


def test_check_cross_claims_config_disabled_returns_empty_and_skips_dispatch(monkeypatch):
    claims = [_claim("claim_a", "numeric", "123"), _claim("claim_b", "numeric", "456")]
    constraints = [_comparison_constraint("<")]

    called = {"comparison": False}

    def fake_check_comparison(*args, **kwargs):
        called["comparison"] = True
        raise AssertionError("check_comparison should not be called when config.enabled is False")

    monkeypatch.setattr("faithfulagent.cross_checker.check_comparison", fake_check_comparison)

    results = check_cross_claims(
        claims,
        constraints,
        [_match_result()],
        CrossClaimCheckConfig(enabled=False, sum_tolerance=0.01),
        "strict",
    )

    assert results == []
    assert called["comparison"] is False


def test_check_cross_claims_dispatches_single_comparison_constraint_pass(monkeypatch):
    claims = [_claim("claim_a", "numeric", "480"), _claim("claim_b", "numeric", "553")]
    constraint = _comparison_constraint("<")
    captured = {}

    def fake_check_comparison(arg_constraint, value_map, mode):
        captured["constraint"] = arg_constraint
        captured["value_map"] = value_map
        captured["mode"] = mode
        return _cross_claim_result("cmp_1", "comparison", "pass")

    monkeypatch.setattr("faithfulagent.cross_checker.check_comparison", fake_check_comparison)

    results = check_cross_claims(
        claims,
        [constraint],
        [],
        CrossClaimCheckConfig(enabled=True, sum_tolerance=0.01),
        "strict",
    )

    assert len(results) == 1
    assert results[0].status == "pass"
    assert captured["constraint"] == constraint
    assert captured["value_map"] == {"claim_a": 480.0, "claim_b": 553.0}
    assert captured["mode"] == "strict"


def test_check_cross_claims_dispatches_single_comparison_constraint_fail(monkeypatch):
    claims = [_claim("claim_a", "numeric", "553"), _claim("claim_b", "numeric", "480")]
    constraint = _comparison_constraint("<")

    def fake_check_comparison(arg_constraint, value_map, mode):
        assert arg_constraint == constraint
        assert value_map == {"claim_a": 553.0, "claim_b": 480.0}
        assert mode == "strict"
        return _cross_claim_result("cmp_1", "comparison", "error")

    monkeypatch.setattr("faithfulagent.cross_checker.check_comparison", fake_check_comparison)

    results = check_cross_claims(
        claims,
        [constraint],
        [],
        CrossClaimCheckConfig(enabled=True, sum_tolerance=0.01),
        "strict",
    )

    assert len(results) == 1
    assert results[0].status == "error"


def test_check_cross_claims_dispatches_single_ranking_constraint(monkeypatch):
    claims = [
        _claim("claim_a", "numeric", "200"),
        _claim("claim_b", "numeric", "150"),
        _claim("claim_c", "numeric", "220"),
    ]
    constraint = _ranking_constraint(ranked_claim_id="claim_b", rank_type="min")
    captured = {}

    def fake_check_ranking(arg_constraint, value_map, mode):
        captured["constraint"] = arg_constraint
        captured["value_map"] = value_map
        captured["mode"] = mode
        return _cross_claim_result("rank_1", "ranking", "pass")

    monkeypatch.setattr("faithfulagent.cross_checker.check_ranking", fake_check_ranking)

    results = check_cross_claims(
        claims,
        [constraint],
        [],
        CrossClaimCheckConfig(enabled=True, sum_tolerance=0.01),
        "strict",
    )

    assert len(results) == 1
    assert results[0].constraint_type == "ranking"
    assert captured["constraint"] == constraint
    assert captured["value_map"] == {"claim_a": 200.0, "claim_b": 150.0, "claim_c": 220.0}
    assert captured["mode"] == "strict"


def test_check_cross_claims_dispatches_single_sum_constraint(monkeypatch):
    claims = [
        _claim("claim_total", "numeric", "850"),
        _claim("claim_flights", "numeric", "340"),
        _claim("claim_hotel", "numeric", "450"),
        _claim("claim_transfers", "numeric", "60"),
    ]
    constraint = _sum_constraint()
    captured = {}

    def fake_check_sum(arg_constraint, value_map, config, mode):
        captured["constraint"] = arg_constraint
        captured["value_map"] = value_map
        captured["config"] = config
        captured["mode"] = mode
        return _cross_claim_result("sum_1", "sum", "pass")

    config = CrossClaimCheckConfig(enabled=True, sum_tolerance=0.01)
    monkeypatch.setattr("faithfulagent.cross_checker.check_sum", fake_check_sum)

    results = check_cross_claims(claims, [constraint], [], config, "strict")

    assert len(results) == 1
    assert results[0].constraint_type == "sum"
    assert captured["constraint"] == constraint
    assert captured["value_map"] == {
        "claim_total": 850.0,
        "claim_flights": 340.0,
        "claim_hotel": 450.0,
        "claim_transfers": 60.0,
    }
    assert captured["config"] == config
    assert captured["mode"] == "strict"


def test_check_cross_claims_dispatches_mixed_constraint_types(monkeypatch):
    claims = [
        _claim("claim_a", "numeric", "480"),
        _claim("claim_b", "numeric", "553"),
        _claim("claim_c", "numeric", "700"),
        _claim("claim_total", "numeric", "1733"),
    ]
    constraints = [
        _comparison_constraint("<"),
        _ranking_constraint(ranked_claim_id="claim_c", rank_type="max", candidate_claim_ids=["claim_a", "claim_b", "claim_c"]),
        _sum_constraint(total_claim_id="claim_total", component_claim_ids=["claim_a", "claim_b", "claim_c"]),
    ]
    calls = []

    def fake_check_comparison(arg_constraint, value_map, mode):
        calls.append(("comparison", arg_constraint.constraint_id, mode, dict(value_map)))
        return _cross_claim_result(arg_constraint.constraint_id, "comparison", "pass")

    def fake_check_ranking(arg_constraint, value_map, mode):
        calls.append(("ranking", arg_constraint.constraint_id, mode, dict(value_map)))
        return _cross_claim_result(arg_constraint.constraint_id, "ranking", "pass")

    def fake_check_sum(arg_constraint, value_map, config, mode):
        calls.append(("sum", arg_constraint.constraint_id, mode, dict(value_map)))
        return _cross_claim_result(arg_constraint.constraint_id, "sum", "pass")

    monkeypatch.setattr("faithfulagent.cross_checker.check_comparison", fake_check_comparison)
    monkeypatch.setattr("faithfulagent.cross_checker.check_ranking", fake_check_ranking)
    monkeypatch.setattr("faithfulagent.cross_checker.check_sum", fake_check_sum)

    results = check_cross_claims(
        claims,
        constraints,
        [],
        CrossClaimCheckConfig(enabled=True, sum_tolerance=0.01),
        "strict",
    )

    assert len(results) == 3
    assert [result.constraint_type for result in results] == ["comparison", "ranking", "sum"]
    assert [call[0] for call in calls] == ["comparison", "ranking", "sum"]


def test_check_cross_claims_string_claim_is_omitted_and_triggers_coercion_failure():
    claims = [
        _claim("claim_num", "numeric", "553"),
        _claim("claim_str", "string", "Delta"),
    ]
    constraint = Constraint(
        constraint_id="cmp_string",
        constraint_type="comparison",
        lhs_claim_id="claim_num",
        operator="<",
        rhs_claim_id="claim_str",
    )

    results = check_cross_claims(
        claims,
        [constraint],
        [],
        CrossClaimCheckConfig(enabled=True, sum_tolerance=0.01),
        "strict",
    )

    assert len(results) == 1
    assert results[0].status == "error"
    assert results[0].diff["coercion_error"] is True
    assert results[0].diff["failed_claim_id"] == "claim_str"
    assert results[0].observed_values == {"claim_num": 553.0, "claim_str": None}


def test_check_cross_claims_currency_claim_is_coerced_to_float(monkeypatch):
    claims = [_claim("claim_a", "currency", "$1,234.56"), _claim("claim_b", "numeric", "2000")]
    constraint = _comparison_constraint("<")
    captured = {}

    def fake_check_comparison(arg_constraint, value_map, mode):
        captured["value_map"] = value_map
        return _cross_claim_result(arg_constraint.constraint_id, "comparison", "pass")

    monkeypatch.setattr("faithfulagent.cross_checker.check_comparison", fake_check_comparison)

    results = check_cross_claims(
        claims,
        [constraint],
        [],
        CrossClaimCheckConfig(enabled=True, sum_tolerance=0.01),
        "strict",
    )

    assert len(results) == 1
    assert captured["value_map"]["claim_a"] == 1234.56
    assert captured["value_map"]["claim_b"] == 2000.0


def test_check_cross_claims_missing_claim_id_triggers_coercion_failure():
    claims = [_claim("claim_a", "numeric", "553")]
    constraint = _comparison_constraint("<")

    results = check_cross_claims(
        claims,
        [constraint],
        [],
        CrossClaimCheckConfig(enabled=True, sum_tolerance=0.01),
        "strict",
    )

    assert len(results) == 1
    assert results[0].status == "error"
    assert results[0].diff["coercion_error"] is True
    assert results[0].diff["failed_claim_id"] == "claim_b"
    assert results[0].observed_values == {"claim_a": 553.0, "claim_b": None}


def test_check_cross_claims_accepts_match_results_parameter_without_error():
    claims = [_claim("claim_a", "numeric", "480"), _claim("claim_b", "numeric", "553")]
    constraint = _comparison_constraint("<")
    match_results = [_match_result()]

    results = check_cross_claims(
        claims,
        [constraint],
        match_results,
        CrossClaimCheckConfig(enabled=True, sum_tolerance=0.01),
        "strict",
    )

    assert len(results) == 1
    assert results[0].status == "pass"
