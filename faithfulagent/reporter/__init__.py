"""Report assembly: aggregates MatchResults into a Report with summary and verdict."""

from faithfulagent.schemas import MatchResult, Report, ReportSummary


def generate_report(match_results: list[MatchResult], mode: str) -> Report:
    total_claims = len(match_results)
    passed = sum(1 for result in match_results if result.status == "pass")
    warnings = sum(1 for result in match_results if result.status == "warning")
    errors = sum(1 for result in match_results if result.status == "error")

    if errors > 0:
        verdict = "fail"
    elif warnings > 0:
        verdict = "warn"
    else:
        verdict = "pass"

    summary = ReportSummary(
        total_claims=total_claims,
        passed=passed,
        warnings=warnings,
        errors=errors,
        verdict=verdict,
    )

    return Report(mode=mode, summary=summary, claims=match_results)
