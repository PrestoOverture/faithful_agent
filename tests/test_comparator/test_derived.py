"""Tests for faithfulagent/comparator/derived.py — Phase 16 coverage."""

from faithfulagent.comparator.derived import check_derived, _collect_numeric_values
from faithfulagent.config import DerivedCheckConfig
from faithfulagent.schemas import ToolCall

CFG = DerivedCheckConfig(tolerance=0.01)
TC  = [ToolCall(call_id="c1", tool_name="t", response={"a": 10, "b": 20, "c": 30})]


def test_sum_detected():
    r = check_derived(60, TC, CFG)
    assert r is not None and r.diff["operation"] == "sum"


def test_average_detected():
    r = check_derived(20, TC, CFG)
    assert r is not None and r.diff["operation"] == "average"


def test_no_match_returns_none():
    assert check_derived(42, TC, CFG) is None


def test_non_numeric_returns_none():
    assert check_derived("abc", TC, CFG) is None


def test_single_value_skipped():
    tc = [ToolCall(call_id="c1", tool_name="t", response={"a": 10})]
    assert check_derived(10, tc, CFG) is None


def test_collect_numeric_values_handles_list():
    acc: list[float] = []
    _collect_numeric_values([1, 2, "three", 4], acc)
    assert acc == [1.0, 2.0, 4.0]


def test_collect_numeric_values_handles_nested_dict():
    acc: list[float] = []
    _collect_numeric_values({"a": {"b": 5}, "c": 10}, acc)
    assert sorted(acc) == [5.0, 10.0]
