"""Tests for faithfulagent/normalizers/openai.py — Phase 16 coverage."""

import pytest
from pydantic import ValidationError

from faithfulagent.normalizers.openai import normalize_openai

BASE = {"id": "c1", "function": {"name": "f", "arguments": "{}"}, "response": {"x": 1}}


def test_json_string_arguments():
    tc = normalize_openai([{**BASE, "function": {"name": "f", "arguments": '{"k": "v"}'}}])
    assert tc[0].request == {"k": "v"}


def test_dict_arguments_passed_through():
    tc = normalize_openai([{**BASE, "function": {"name": "f", "arguments": {"k": "v"}}}])
    assert tc[0].request == {"k": "v"}


def test_invalid_json_arguments_fallback():
    tc = normalize_openai([{**BASE, "function": {"name": "f", "arguments": "not-json"}}])
    assert tc[0].request == {}


def test_none_arguments_fallback():
    tc = normalize_openai([{**BASE, "function": {"name": "f", "arguments": None}}])
    assert tc[0].request == {}


def test_missing_id_raises():
    with pytest.raises(ValueError):
        normalize_openai([{"function": BASE["function"], "response": {}}])


def test_missing_function_raises():
    with pytest.raises(ValueError):
        normalize_openai([{"id": "c1", "response": {}}])


def test_missing_response_raises():
    with pytest.raises((ValidationError, KeyError)):
        normalize_openai([{"id": "c1", "function": {"name": "f", "arguments": "{}"}}])
