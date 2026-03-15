"""Comparator entry point: matches claims to tool call fields and runs checks."""

import re

from faithfulagent.comparator.matcher import match_claim
from faithfulagent.comparator.numeric import check_numeric
from faithfulagent.comparator.string import check_string
from faithfulagent.config import VerifyConfig
from faithfulagent.schemas import Claim, MatchResult, ToolCall

SYMBOL_MAP = {"$": "USD", "¥": "CNY", "€": "EUR", "£": "GBP"}


def _extract_currency_symbol(value: str) -> str | None:
    symbols = re.findall(r"[\$\¥\€\£]", value)
    if not symbols:
        return None
    return symbols[0]


def compare(
    claims: list[Claim], tool_calls: list[ToolCall], config: VerifyConfig
) -> list[MatchResult]:
    results: list[MatchResult] = []

    for claim in claims:
        matched_call_id, matched_field, expected_value = match_claim(
            claim, tool_calls
        )

        if matched_call_id is None:
            results.append(
                MatchResult(
                    claim_id=claim.claim_id,
                    status="warning",
                    mismatch_category="unsupported",
                    severity="warning",
                    match_confidence="low",
                )
            )
            continue

        if claim.claim_type == "numeric":
            result = check_numeric(
                claim.extracted_value, expected_value, config.numeric
            )
        elif claim.claim_type in {"string", "date"}:
            result = check_string(
                claim.extracted_value, expected_value, config.string
            )
        elif claim.claim_type == "currency":
            numeric_result = check_numeric(
                claim.extracted_value, expected_value, config.numeric
            )
            observed_symbol = _extract_currency_symbol(str(claim.extracted_value))
            expected_symbol = _extract_currency_symbol(str(expected_value))
            observed_code = SYMBOL_MAP.get(observed_symbol) if observed_symbol else None
            expected_code = SYMBOL_MAP.get(expected_symbol) if expected_symbol else None

            if (
                observed_code
                and expected_code
                and observed_code != expected_code
            ):
                numeric_result.status = "error"
                numeric_result.mismatch_category = "unit_conversion"
                numeric_result.severity = "error"
            result = numeric_result
        else:
            result = MatchResult(
                claim_id="",
                status="warning",
                mismatch_category="unsupported",
                severity="warning",
            )

        result.claim_id = claim.claim_id
        result.matched_call_id = matched_call_id
        result.matched_field = matched_field
        result.expected_value = expected_value
        result.observed_value = claim.extracted_value
        result.match_confidence = "high" if claim.source_ref else "medium"

        results.append(result)

    return results
