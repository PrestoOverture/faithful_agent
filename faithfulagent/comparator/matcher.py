"""Claim-to-tool-field matching logic for strict and heuristic modes."""

from typing import Any, Optional

from jsonpath_ng import parse as jp_parse
from rapidfuzz import fuzz

from faithfulagent.schemas import Claim, ToolCall


def _flatten_response(call_id: str, response: dict) -> list[tuple[str, str, Any]]:
    flat: list[tuple[str, str, Any]] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{path}.{key}")
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
            return
        if isinstance(value, (str, int, float, bool)) or value is None:
            flat.append((call_id, path, value))

    walk(response, "$")
    return flat


def match_claim(
    claim: Claim, tool_calls: list[ToolCall]
) -> tuple[Optional[str], Optional[str], Any]:
    if claim.source_ref is not None:
        tool_call_map = {call.call_id: call for call in tool_calls}
        tool_call = tool_call_map[claim.source_ref.call_id]
        matches = jp_parse(claim.source_ref.path).find(tool_call.response)
        resolved = matches[0].value if matches else None
        return tool_call.call_id, claim.source_ref.path, resolved

    flat: list[tuple[str, str, Any]] = []
    for tool_call in tool_calls:
        flat.extend(_flatten_response(tool_call.call_id, tool_call.response))

    call_recency = {call.call_id: index for index, call in enumerate(tool_calls)}

    if claim.claim_type == "numeric":
        raw = str(claim.extracted_value)
        cleaned = (
            raw.replace(",", "")
            .replace("$", "")
            .replace("¥", "")
            .replace("€", "")
            .replace("£", "")
        )
        try:
            target = float(cleaned)
        except ValueError:
            return None, None, None

        numeric_entries = [
            entry for entry in flat if isinstance(entry[2], (int, float))
        ]
        if not numeric_entries:
            return None, None, None

        best_entry: Optional[tuple[str, str, Any]] = None
        best_diff = float("inf")
        best_recency = -1

        for call_id, path, value in numeric_entries:
            diff = abs(float(value) - target)
            recency = call_recency.get(call_id, -1)
            if diff < best_diff or (diff == best_diff and recency > best_recency):
                best_entry = (call_id, path, value)
                best_diff = diff
                best_recency = recency

        if best_entry is None:
            return None, None, None
        return best_entry

    if claim.claim_type in {"string", "date", "currency"}:
        string_entries = [entry for entry in flat if isinstance(entry[2], str)]
        if not string_entries:
            return None, None, None

        target = str(claim.extracted_value)
        best_entry = None
        best_score = -1
        best_recency = -1

        for call_id, path, value in string_entries:
            score = fuzz.ratio(target, str(value))
            recency = call_recency.get(call_id, -1)
            if score > best_score or (score == best_score and recency > best_recency):
                best_entry = (call_id, path, value)
                best_score = score
                best_recency = recency

        if best_entry is None or best_score < 60:
            return None, None, None
        return best_entry

    return None, None, None
