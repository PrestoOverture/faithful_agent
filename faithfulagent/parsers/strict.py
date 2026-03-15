"""Strict mode parser: extracts claims from structured agent annotations."""

from typing import Any

from jsonpath_ng import parse as jp_parse

from faithfulagent.schemas import Claim, SourceRef, TextSpan, ToolCall


def parse_strict(agent_output: dict, tool_calls: list[ToolCall]) -> list[Claim]:
    text = agent_output["text"]
    refs = agent_output.get("refs", [])
    claims: list[Claim] = []

    tool_call_map = {call.call_id: call for call in tool_calls}

    for i, ref in enumerate(refs):
        call_id = ref["call_id"]
        tool_call = tool_call_map.get(call_id)
        if tool_call is None:
            raise ValueError(f"No tool call found with call_id '{call_id}'")

        span = ref["span"]
        span_text = text[span[0] : span[1]]

        matches = jp_parse(ref["path"]).find(tool_call.response)
        resolved: Any = matches[0].value if matches else None

        if isinstance(resolved, (int, float)):
            claim_type = "numeric"
        elif isinstance(resolved, str):
            claim_type = "string"
        else:
            claim_type = "string"

        claims.append(
            Claim(
                claim_id=f"claim_{i}",
                claim_type=claim_type,
                text_span=TextSpan(start=span[0], end=span[1], text=span_text),
                extracted_value=span_text,
                source_ref=SourceRef(
                    call_id=call_id,
                    path=ref["path"],
                    transform=ref.get("transform", "none"),
                ),
            )
        )

    return claims
