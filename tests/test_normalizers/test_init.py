"""Tests for faithfulagent/normalizers/__init__.py — Phase 4, Step 4.2."""

import pytest

from faithfulagent.normalizers import normalize
from faithfulagent.schemas import ToolCall

RAW_DATA = [{"call_id": "c1", "tool_name": "get_price", "response": {"price": 553}}]


def test_raw_format_delegates_and_returns_tool_calls():
    result = normalize(RAW_DATA, "raw")
    assert len(result) == 1
    assert isinstance(result[0], ToolCall)
    assert result[0].call_id == "c1"


def test_raw_format_is_default():
    result = normalize(RAW_DATA)
    assert isinstance(result[0], ToolCall)


def test_openai_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        normalize(RAW_DATA, "openai")


def test_langchain_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        normalize(RAW_DATA, "langchain")


def test_unknown_format_raises_value_error():
    with pytest.raises(ValueError):
        normalize(RAW_DATA, "unknown")


def test_unknown_format_error_message():
    with pytest.raises(ValueError, match="foo"):
        normalize(RAW_DATA, "foo")
