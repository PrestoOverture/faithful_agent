"""Report formatters: JSON serialization and human-readable summary."""

from faithfulagent.schemas import Report


def to_json(report: Report) -> str:
    return report.model_dump_json(indent=2)


def to_summary(report: Report) -> str:
    summary = report.summary
    total = summary.total_claims
    if summary.verdict == "pass":
        return f"PASS: {total}/{total} claims verified"
    return (
        f"{summary.verdict.upper()}: "
        f"{summary.passed} passed, {summary.warnings} warning(s), "
        f"{summary.errors} error(s) out of {total} claims"
    )
