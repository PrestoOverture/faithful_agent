"""Normalizer dispatcher: routes input to format-specific normalizer."""

from faithfulagent.normalizers.langchain import normalize_langchain
from faithfulagent.normalizers.openai import normalize_openai
from faithfulagent.normalizers.raw import normalize_raw
from faithfulagent.schemas import ToolCall


def normalize(tool_calls: list[dict], source_format: str = "raw") -> list[ToolCall]:
    if source_format == "raw":
        return normalize_raw(tool_calls)
    if source_format in {"openai", "langchain"}:
        if source_format == "openai":
            return normalize_openai(tool_calls)
        return normalize_langchain(tool_calls)
    raise ValueError(f"Unknown source_format: '{source_format}'")
