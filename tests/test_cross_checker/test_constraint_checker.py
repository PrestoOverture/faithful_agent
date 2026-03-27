import importlib

import pytest

from faithfulagent.config import CrossClaimCheckConfig
from faithfulagent.cross_checker.constraint_checker import (
    check_comparison,
    check_ranking,
    check_sum,
)
from faithfulagent.schemas import Constraint


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
