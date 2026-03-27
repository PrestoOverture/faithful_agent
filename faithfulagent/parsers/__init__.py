"""Parser dispatcher: routes agent output to strict or heuristic parser."""

from faithfulagent.parsers.heuristic import parse_heuristic
from faithfulagent.parsers.strict import parse_strict
from faithfulagent.schemas import Claim, Constraint, ToolCall


def parse(
    agent_output, tool_calls: list[ToolCall], mode: str = "strict"
) -> tuple[list[Claim], list[Constraint]]:
    if mode == "strict":
        return parse_strict(agent_output, tool_calls)
    if mode == "heuristic":
        return parse_heuristic(agent_output, tool_calls)
    raise ValueError(f"Unknown mode: '{mode}'")
