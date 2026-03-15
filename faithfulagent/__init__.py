from faithfulagent.schemas import (
    Claim,
    MatchResult,
    Report,
    ReportSummary,
    SourceRef,
    TextSpan,
    ToolCall,
)
from faithfulagent.config import VerifyConfig
from faithfulagent.normalizers import normalize
from faithfulagent.parsers import parse
from faithfulagent.comparator import compare
from faithfulagent.reporter import generate_report


def verify(
    agent_output: str | dict,
    tool_calls: list[dict],
    mode: str = "strict",
    source_format: str = "raw",
    checks: dict | None = None,
) -> Report:
    config = VerifyConfig.from_dict(checks or {})
    normalized = normalize(tool_calls, source_format)
    claims = parse(agent_output, normalized, mode)
    results = compare(claims, normalized, config)
    report = generate_report(results, mode)
    return report

__all__ = [
    "Claim",
    "MatchResult",
    "Report",
    "ReportSummary",
    "SourceRef",
    "TextSpan",
    "ToolCall",
    "verify",
]
