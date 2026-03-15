"""OpenAI tool call normalizer."""

import json

from faithfulagent.schemas import ToolCall


def _parse_arguments(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def normalize_openai(tool_calls: list[dict]) -> list[ToolCall]:
    normalized: list[ToolCall] = []
    for item in tool_calls:
        if "id" not in item or "function" not in item:
            raise ValueError("OpenAI tool call must include 'id' and 'function'")
        function = item["function"]
        normalized.append(
            ToolCall(
                call_id=item["id"],
                tool_name=function["name"],
                request=_parse_arguments(function.get("arguments")),
                response=item["response"],
                timestamp=item.get("timestamp"),
            )
        )
    return normalized
