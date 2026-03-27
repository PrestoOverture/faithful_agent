"""Tests for faithfulagent/parsers/strict.py — Phase 5, Step 5.1."""

import pytest

from faithfulagent.parsers.strict import parse_strict
from faithfulagent.schemas import Constraint, ToolCall


TOOL_CALL = ToolCall(
    call_id="call_1",
    tool_name="get_flight",
    response={"price": 553, "airline": "Delta"},
)

AGENT_OUTPUT = {
    "text": "It costs 553 on Delta",
    "refs": [
        {"span": [9, 12], "call_id": "call_1", "path": "$.price"},
        {"span": [16, 21], "call_id": "call_1", "path": "$.airline"},
    ],
}


def test_returns_two_claims():
    claims, _ = parse_strict(AGENT_OUTPUT, [TOOL_CALL])
    assert len(claims) == 2


def test_numeric_claim():
    claims, _ = parse_strict(AGENT_OUTPUT, [TOOL_CALL])
    c = claims[0]
    assert c.claim_type == "numeric"
    assert c.extracted_value == "553"
    assert c.source_ref.path == "$.price"
    assert c.text_span.start == 9
    assert c.text_span.end == 12
    assert c.text_span.text == "553"


def test_string_claim():
    claims, _ = parse_strict(AGENT_OUTPUT, [TOOL_CALL])
    c = claims[1]
    assert c.claim_type == "string"
    assert c.extracted_value == "Delta"
    assert c.source_ref.path == "$.airline"
    assert c.text_span.start == 16
    assert c.text_span.end == 21
    assert c.text_span.text == "Delta"


def test_claim_ids_auto_generated():
    claims, _ = parse_strict(AGENT_OUTPUT, [TOOL_CALL])
    assert claims[0].claim_id == "claim_0"
    assert claims[1].claim_id == "claim_1"


def test_source_ref_call_id():
    claims, _ = parse_strict(AGENT_OUTPUT, [TOOL_CALL])
    assert claims[0].source_ref.call_id == "call_1"


def test_transform_defaults_to_none():
    claims, _ = parse_strict(AGENT_OUTPUT, [TOOL_CALL])
    assert claims[0].source_ref.transform == "none"


def test_transform_explicit():
    output = {
        "text": "About 553",
        "refs": [{"span": [6, 9], "call_id": "call_1", "path": "$.price", "transform": "round"}],
    }
    claims, _ = parse_strict(output, [TOOL_CALL])
    assert claims[0].source_ref.transform == "round"


def test_unknown_call_id_raises():
    output = {
        "text": "costs 553",
        "refs": [{"span": [6, 9], "call_id": "bad_id", "path": "$.price"}],
    }
    with pytest.raises(ValueError, match="bad_id"):
        parse_strict(output, [TOOL_CALL])


def test_nested_jsonpath():
    tc = ToolCall(
        call_id="call_2",
        tool_name="get_data",
        response={"data": {"name": "JFK"}},
    )
    output = {
        "text": "Departs from JFK",
        "refs": [{"span": [13, 16], "call_id": "call_2", "path": "$.data.name"}],
    }
    claims, _ = parse_strict(output, [tc])
    assert claims[0].claim_type == "string"
    assert claims[0].extracted_value == "JFK"
    assert claims[0].source_ref.path == "$.data.name"


def test_missing_jsonpath_field_gives_string_type():
    output = {
        "text": "costs 999",
        "refs": [{"span": [6, 9], "call_id": "call_1", "path": "$.nonexistent"}],
    }
    claims, _ = parse_strict(output, [TOOL_CALL])
    assert claims[0].claim_type == "string"
    assert claims[0].extracted_value == "999"


def test_empty_refs_returns_empty_list():
    output = {"text": "No refs here", "refs": []}
    claims, constraints = parse_strict(output, [TOOL_CALL])
    assert claims == []
    assert constraints == []


def test_multiple_tool_calls():
    tc2 = ToolCall(call_id="call_2", tool_name="get_hotel", response={"hotel": "Marriott"})
    output = {
        "text": "Fly Delta, stay at Marriott",
        "refs": [
            {"span": [4, 9], "call_id": "call_1", "path": "$.airline"},
            {"span": [19, 27], "call_id": "call_2", "path": "$.hotel"},
        ],
    }
    claims, _ = parse_strict(output, [TOOL_CALL, tc2])
    assert claims[0].extracted_value == "Delta"
    assert claims[1].extracted_value == "Marriott"
    assert claims[0].source_ref.call_id == "call_1"
    assert claims[1].source_ref.call_id == "call_2"


def test_no_constraints_key_returns_empty_constraint_list():
    claims, constraints = parse_strict(AGENT_OUTPUT, [TOOL_CALL])
    assert len(claims) == 2
    assert constraints == []


def test_comparison_constraint_maps_ref_indices_to_claim_ids():
    output = {
        "text": "Flight A costs 553. Flight B costs 600.",
        "refs": [
            {"span": [15, 18], "call_id": "call_cmp", "path": "$.flight_a_price"},
            {"span": [35, 38], "call_id": "call_cmp", "path": "$.flight_b_price"},
        ],
        "constraints": [
            {
                "constraint_id": "cmp_1",
                "constraint_type": "comparison",
                "lhs_ref_index": 0,
                "operator": "<",
                "rhs_ref_index": 1,
            }
        ],
    }
    tool_call = ToolCall(
        call_id="call_cmp",
        tool_name="get_flights",
        response={"flight_a_price": 553, "flight_b_price": 600},
    )

    claims, constraints = parse_strict(output, [tool_call])

    assert len(claims) == 2
    assert len(constraints) == 1
    assert isinstance(constraints[0], Constraint)
    assert constraints[0].constraint_type == "comparison"
    assert constraints[0].constraint_id == "cmp_1"
    assert constraints[0].lhs_claim_id == claims[0].claim_id
    assert constraints[0].rhs_claim_id == claims[1].claim_id
    assert constraints[0].operator == "<"


def test_ranking_constraint_maps_ref_indices_to_candidate_claim_ids():
    output = {
        "text": "Hotel A 200. Hotel B 150. Hotel C 220.",
        "refs": [
            {"span": [8, 11], "call_id": "call_rank", "path": "$.hotel_a_price"},
            {"span": [21, 24], "call_id": "call_rank", "path": "$.hotel_b_price"},
            {"span": [34, 37], "call_id": "call_rank", "path": "$.hotel_c_price"},
        ],
        "constraints": [
            {
                "constraint_id": "rank_1",
                "constraint_type": "ranking",
                "ranked_ref_index": 0,
                "rank_type": "min",
                "candidate_ref_indices": [0, 1, 2],
            }
        ],
    }
    tool_call = ToolCall(
        call_id="call_rank",
        tool_name="get_hotels",
        response={"hotel_a_price": 200, "hotel_b_price": 150, "hotel_c_price": 220},
    )

    claims, constraints = parse_strict(output, [tool_call])

    assert len(claims) == 3
    assert len(constraints) == 1
    assert constraints[0].constraint_type == "ranking"
    assert constraints[0].constraint_id == "rank_1"
    assert constraints[0].ranked_claim_id == claims[0].claim_id
    assert constraints[0].rank_type == "min"
    assert constraints[0].candidate_claim_ids == [claim.claim_id for claim in claims]


def test_sum_constraint_maps_ref_indices_to_total_and_component_claim_ids():
    output = {
        "text": "Flights 340. Hotel 450. Transfers 60. Total 850.",
        "refs": [
            {"span": [8, 11], "call_id": "call_sum", "path": "$.flights"},
            {"span": [19, 22], "call_id": "call_sum", "path": "$.hotel"},
            {"span": [34, 36], "call_id": "call_sum", "path": "$.transfers"},
            {"span": [44, 47], "call_id": "call_sum", "path": "$.total"},
        ],
        "constraints": [
            {
                "constraint_id": "sum_1",
                "constraint_type": "sum",
                "total_ref_index": 3,
                "component_ref_indices": [0, 1, 2],
            }
        ],
    }
    tool_call = ToolCall(
        call_id="call_sum",
        tool_name="get_trip_breakdown",
        response={"flights": 340, "hotel": 450, "transfers": 60, "total": 850},
    )

    claims, constraints = parse_strict(output, [tool_call])

    assert len(claims) == 4
    assert len(constraints) == 1
    assert constraints[0].constraint_type == "sum"
    assert constraints[0].constraint_id == "sum_1"
    assert constraints[0].total_claim_id == claims[3].claim_id
    assert constraints[0].component_claim_ids == [claims[0].claim_id, claims[1].claim_id, claims[2].claim_id]


def test_out_of_bounds_constraint_ref_index_raises_value_error():
    output = {
        "text": "Flight A costs 553. Flight B costs 600.",
        "refs": [
            {"span": [15, 18], "call_id": "call_cmp", "path": "$.flight_a_price"},
            {"span": [35, 38], "call_id": "call_cmp", "path": "$.flight_b_price"},
        ],
        "constraints": [
            {
                "constraint_id": "cmp_1",
                "constraint_type": "comparison",
                "lhs_ref_index": 99,
                "operator": "<",
                "rhs_ref_index": 1,
            }
        ],
    }
    tool_call = ToolCall(
        call_id="call_cmp",
        tool_name="get_flights",
        response={"flight_a_price": 553, "flight_b_price": 600},
    )

    with pytest.raises(ValueError):
        parse_strict(output, [tool_call])


def test_negative_constraint_ref_index_raises_value_error():
    output = {
        "text": "Flight A costs 553. Flight B costs 600.",
        "refs": [
            {"span": [15, 18], "call_id": "call_cmp", "path": "$.flight_a_price"},
            {"span": [35, 38], "call_id": "call_cmp", "path": "$.flight_b_price"},
        ],
        "constraints": [
            {
                "constraint_id": "cmp_1",
                "constraint_type": "comparison",
                "lhs_ref_index": -1,
                "operator": "<",
                "rhs_ref_index": 1,
            }
        ],
    }
    tool_call = ToolCall(
        call_id="call_cmp",
        tool_name="get_flights",
        response={"flight_a_price": 553, "flight_b_price": 600},
    )

    with pytest.raises(ValueError):
        parse_strict(output, [tool_call])


def test_duplicate_constraint_id_raises_value_error():
    output = {
        "text": "Flight A costs 553. Flight B costs 600.",
        "refs": [
            {"span": [15, 18], "call_id": "call_cmp", "path": "$.flight_a_price"},
            {"span": [35, 38], "call_id": "call_cmp", "path": "$.flight_b_price"},
        ],
        "constraints": [
            {
                "constraint_id": "dup_1",
                "constraint_type": "comparison",
                "lhs_ref_index": 0,
                "operator": "<",
                "rhs_ref_index": 1,
            },
            {
                "constraint_id": "dup_1",
                "constraint_type": "comparison",
                "lhs_ref_index": 1,
                "operator": ">",
                "rhs_ref_index": 0,
            },
        ],
    }
    tool_call = ToolCall(
        call_id="call_cmp",
        tool_name="get_flights",
        response={"flight_a_price": 553, "flight_b_price": 600},
    )

    with pytest.raises(ValueError):
        parse_strict(output, [tool_call])


def test_constraint_ref_pointing_to_string_claim_raises_value_error():
    output = {
        "text": "It costs 553 on Delta",
        "refs": [
            {"span": [9, 12], "call_id": "call_1", "path": "$.price"},
            {"span": [16, 21], "call_id": "call_1", "path": "$.airline"},
        ],
        "constraints": [
            {
                "constraint_id": "cmp_1",
                "constraint_type": "comparison",
                "lhs_ref_index": 0,
                "operator": "<",
                "rhs_ref_index": 1,
            }
        ],
    }

    with pytest.raises(ValueError):
        parse_strict(output, [TOOL_CALL])


def test_multiple_constraints_are_all_returned():
    output = {
        "text": "Hotel A 200. Hotel B 150. Hotel C 220. Total 570.",
        "refs": [
            {"span": [8, 11], "call_id": "call_multi", "path": "$.hotel_a_price"},
            {"span": [21, 24], "call_id": "call_multi", "path": "$.hotel_b_price"},
            {"span": [34, 37], "call_id": "call_multi", "path": "$.hotel_c_price"},
            {"span": [45, 48], "call_id": "call_multi", "path": "$.total"},
        ],
        "constraints": [
            {
                "constraint_id": "rank_1",
                "constraint_type": "ranking",
                "ranked_ref_index": 1,
                "rank_type": "min",
                "candidate_ref_indices": [0, 1, 2],
            },
            {
                "constraint_id": "sum_1",
                "constraint_type": "sum",
                "total_ref_index": 3,
                "component_ref_indices": [0, 1, 2],
            },
        ],
    }
    tool_call = ToolCall(
        call_id="call_multi",
        tool_name="get_hotels",
        response={"hotel_a_price": 200, "hotel_b_price": 150, "hotel_c_price": 220, "total": 570},
    )

    claims, constraints = parse_strict(output, [tool_call])

    assert len(claims) == 4
    assert len(constraints) == 2
    assert [constraint.constraint_id for constraint in constraints] == ["rank_1", "sum_1"]
