"""Raw JSON normalizer: validates and converts raw dict input to ToolCall objects."""

from faithfulagent.schemas import ToolCall


def normalize_raw(tool_calls: list[dict]) -> list[ToolCall]:
    return [ToolCall(**item) for item in tool_calls]
