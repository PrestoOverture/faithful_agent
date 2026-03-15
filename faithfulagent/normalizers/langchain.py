"""LangChain tool call normalizer."""

from faithfulagent.schemas import ToolCall


def normalize_langchain(tool_calls: list[dict]) -> list[ToolCall]:
    normalized: list[ToolCall] = []
    for item in tool_calls:
        call_id = item.get("tool_call_id") or item.get("id")
        if call_id is None:
            raise ValueError("LangChain tool call must include 'tool_call_id' or 'id'")
        normalized.append(
            ToolCall(
                call_id=call_id,
                tool_name=item["tool"],
                request=item.get("tool_input", {}),
                response=item["output"],
                timestamp=item.get("timestamp"),
            )
        )
    return normalized
