"""Normalizer dispatcher: routes input to format-specific normalizer."""

from faithfulagent.normalizers.raw import normalize_raw
from faithfulagent.schemas import ToolCall


def normalize(tool_calls: list[dict], source_format: str = "raw") -> list[ToolCall]:
    if source_format == "raw":
        return normalize_raw(tool_calls)
    if source_format in {"openai", "langchain"}:
        raise NotImplementedError(
            f"'{source_format}' normalizer is not yet implemented"
        )
    raise ValueError(f"Unknown source_format: '{source_format}'")
