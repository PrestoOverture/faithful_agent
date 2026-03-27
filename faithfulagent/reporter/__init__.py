"""Report assembly: aggregates MatchResults into a Report with summary and verdict."""

from faithfulagent.schemas import CrossClaimResult, MatchResult, Report, ReportSummary


def generate_report(
    match_results: list[MatchResult],
    mode: str,
    cross_claim_results: list[CrossClaimResult] | None = None,
) -> Report:
    if cross_claim_results is None:
        cross_claim_results = []
    total_claims = len(match_results)
    passed = sum(1 for result in match_results if result.status == "pass")
    warnings = sum(1 for result in match_results if result.status == "warning")
    errors = sum(1 for result in match_results if result.status == "error")
    cross_claim_violations = sum(
        1 for result in cross_claim_results if result.status == "error"
    )
    has_cross_claim_error = any(
        result.status == "error" and result.severity == "error"
        for result in cross_claim_results
    )
    has_cross_claim_warning = any(
        result.status == "error" and result.severity == "warning"
        for result in cross_claim_results
    )

    if errors > 0 or has_cross_claim_error:
        verdict = "fail"
    elif warnings > 0 or has_cross_claim_warning:
        verdict = "warn"
    else:
        verdict = "pass"

    summary = ReportSummary(
        total_claims=total_claims,
        passed=passed,
        warnings=warnings,
        errors=errors,
        cross_claim_violations=cross_claim_violations,
        verdict=verdict,
    )

    return Report(
        mode=mode,
        summary=summary,
        claims=match_results,
        cross_claim_results=cross_claim_results,
    )
