"""Tests for faithfulagent/parsers/strict.py — Phase 5, Step 5.1."""

import pytest

from faithfulagent.parsers.strict import parse_strict
from faithfulagent.schemas import ToolCall


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
    claims = parse_strict(AGENT_OUTPUT, [TOOL_CALL])
    assert len(claims) == 2


def test_numeric_claim():
    claims = parse_strict(AGENT_OUTPUT, [TOOL_CALL])
    c = claims[0]
    assert c.claim_type == "numeric"
    assert c.extracted_value == "553"
    assert c.source_ref.path == "$.price"
    assert c.text_span.start == 9
    assert c.text_span.end == 12
    assert c.text_span.text == "553"


def test_string_claim():
    claims = parse_strict(AGENT_OUTPUT, [TOOL_CALL])
    c = claims[1]
    assert c.claim_type == "string"
    assert c.extracted_value == "Delta"
    assert c.source_ref.path == "$.airline"
    assert c.text_span.start == 16
    assert c.text_span.end == 21
    assert c.text_span.text == "Delta"


def test_claim_ids_auto_generated():
    claims = parse_strict(AGENT_OUTPUT, [TOOL_CALL])
    assert claims[0].claim_id == "claim_0"
    assert claims[1].claim_id == "claim_1"


def test_source_ref_call_id():
    claims = parse_strict(AGENT_OUTPUT, [TOOL_CALL])
    assert claims[0].source_ref.call_id == "call_1"


def test_transform_defaults_to_none():
    claims = parse_strict(AGENT_OUTPUT, [TOOL_CALL])
    assert claims[0].source_ref.transform == "none"


def test_transform_explicit():
    output = {
        "text": "About 553",
        "refs": [{"span": [6, 9], "call_id": "call_1", "path": "$.price", "transform": "round"}],
    }
    claims = parse_strict(output, [TOOL_CALL])
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
    claims = parse_strict(output, [tc])
    assert claims[0].claim_type == "string"
    assert claims[0].extracted_value == "JFK"
    assert claims[0].source_ref.path == "$.data.name"


def test_missing_jsonpath_field_gives_string_type():
    output = {
        "text": "costs 999",
        "refs": [{"span": [6, 9], "call_id": "call_1", "path": "$.nonexistent"}],
    }
    claims = parse_strict(output, [TOOL_CALL])
    assert claims[0].claim_type == "string"
    assert claims[0].extracted_value == "999"


def test_empty_refs_returns_empty_list():
    output = {"text": "No refs here", "refs": []}
    assert parse_strict(output, [TOOL_CALL]) == []


def test_multiple_tool_calls():
    tc2 = ToolCall(call_id="call_2", tool_name="get_hotel", response={"hotel": "Marriott"})
    output = {
        "text": "Fly Delta, stay at Marriott",
        "refs": [
            {"span": [4, 9], "call_id": "call_1", "path": "$.airline"},
            {"span": [19, 27], "call_id": "call_2", "path": "$.hotel"},
        ],
    }
    claims = parse_strict(output, [TOOL_CALL, tc2])
    assert claims[0].extracted_value == "Delta"
    assert claims[1].extracted_value == "Marriott"
    assert claims[0].source_ref.call_id == "call_1"
    assert claims[1].source_ref.call_id == "call_2"
