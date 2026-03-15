"""Parser dispatcher: routes agent output to strict or heuristic parser."""

from faithfulagent.parsers.strict import parse_strict
from faithfulagent.schemas import Claim, ToolCall


def parse(agent_output, tool_calls: list[ToolCall], mode: str = "strict") -> list[Claim]:
    if mode == "strict":
        return parse_strict(agent_output, tool_calls)
    if mode == "heuristic":
        raise NotImplementedError("'heuristic' parser is not yet implemented")
    raise ValueError(f"Unknown mode: '{mode}'")
