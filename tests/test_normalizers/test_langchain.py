"""Tests for faithfulagent/normalizers/langchain.py — Phase 16 coverage."""

import pytest
from pydantic import ValidationError

from faithfulagent.normalizers.langchain import normalize_langchain

BASE = {"tool": "f", "tool_call_id": "c1", "output": {"x": 1}}


def test_basic_normalization():
    tc = normalize_langchain([BASE])[0]
    assert tc.call_id == "c1" and tc.tool_name == "f"


def test_id_fallback():
    tc = normalize_langchain([{"tool": "f", "id": "c2", "output": {"x": 1}}])[0]
    assert tc.call_id == "c2"


def test_missing_call_id_raises():
    with pytest.raises(ValueError):
        normalize_langchain([{"tool": "f", "output": {"x": 1}}])


def test_missing_tool_input_defaults_to_empty():
    tc = normalize_langchain([BASE])[0]
    assert tc.request == {}


def test_missing_output_raises():
    with pytest.raises((ValidationError, KeyError)):
        normalize_langchain([{"tool": "f", "tool_call_id": "c1"}])
