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
from faithfulagent.cross_checker import check_cross_claims
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
    claims, constraints = parse(agent_output, normalized, mode)
    match_results = compare(claims, normalized, config)
    cross_claim_results = check_cross_claims(
        claims, constraints, match_results, config.cross_claim, mode
    )
    report = generate_report(match_results, mode, cross_claim_results)
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
