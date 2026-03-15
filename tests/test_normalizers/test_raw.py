"""Tests for faithfulagent/normalizers/raw.py — Phase 4, Step 4.1."""

import pytest
from pydantic import ValidationError

from faithfulagent.normalizers.raw import normalize_raw
from faithfulagent.schemas import ToolCall


def test_single_tool_call():
    result = normalize_raw([
        {"call_id": "c1", "tool_name": "get_price", "response": {"price": 553}}
    ])
    assert len(result) == 1
    assert isinstance(result[0], ToolCall)
    assert result[0].call_id == "c1"
    assert result[0].tool_name == "get_price"
    assert result[0].response == {"price": 553}
    assert result[0].request == {}
    assert result[0].timestamp is None


def test_two_tool_calls():
    result = normalize_raw([
        {"call_id": "c1", "tool_name": "get_price", "response": {"price": 553}},
        {"call_id": "c2", "tool_name": "get_weather", "response": {"temp": 72}},
    ])
    assert len(result) == 2
    assert result[0].call_id == "c1"
    assert result[1].call_id == "c2"


def test_optional_fields_preserved():
    from datetime import datetime, timezone
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    result = normalize_raw([{
        "call_id": "c1",
        "tool_name": "get_price",
        "request": {"q": "NYC"},
        "response": {"price": 553},
        "timestamp": ts,
    }])
    assert result[0].request == {"q": "NYC"}
    assert result[0].timestamp == ts


def test_nested_response_preserved():
    result = normalize_raw([{
        "call_id": "c1",
        "tool_name": "get_data",
        "response": {"data": {"price": 553, "currency": "CNY"}},
    }])
    assert result[0].response == {"data": {"price": 553, "currency": "CNY"}}


def test_missing_response_raises():
    with pytest.raises(ValidationError):
        normalize_raw([{"call_id": "c1", "tool_name": "get_price"}])


def test_missing_call_id_raises():
    with pytest.raises(ValidationError):
        normalize_raw([{"tool_name": "get_price", "response": {"price": 553}}])


def test_missing_tool_name_raises():
    with pytest.raises(ValidationError):
        normalize_raw([{"call_id": "c1", "response": {"price": 553}}])


def test_empty_list_returns_empty():
    assert normalize_raw([]) == []
