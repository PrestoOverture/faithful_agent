"""Heuristic parser: regex + optional spaCy NER claim extraction."""

import re

from faithfulagent.schemas import Claim, TextSpan, ToolCall

CURRENCY_SYMBOLS = {"$", "¥", "€", "£"}
NUMBER_PATTERN = re.compile(r"[¥$€£]\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?")

# Patterns for time and date strings whose digit components should not be
# extracted as standalone numeric claims.
_TIME_PATTERN = re.compile(
    r"\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?",
)
_ISO_DATE_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?",
)


def _time_date_spans(text: str) -> list[tuple[int, int]]:
    """Return character spans occupied by time or ISO-date patterns."""
    spans: list[tuple[int, int]] = []
    for pattern in (_ISO_DATE_PATTERN, _TIME_PATTERN):
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end()))
    return spans

_nlp = None
_SPACY_ERROR = (
    "spaCy and en_core_web_sm are required for heuristic mode. Install with: "
    "pip install faithfulagent[heuristic] && python -m spacy download en_core_web_sm"
)


def _load_spacy_model():
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy

        _nlp = spacy.load("en_core_web_sm")
        return _nlp
    except (ImportError, OSError) as exc:
        raise ImportError(_SPACY_ERROR) from exc


def parse_heuristic(agent_output: str, tool_calls: list[ToolCall]) -> list[Claim]:
    regex_claims: list[Claim] = []
    claim_index = 0

    time_date_spans = _time_date_spans(agent_output)

    for match in NUMBER_PATTERN.finditer(agent_output):
        m_start, m_end = match.start(), match.end()
        if any(
            not (td_end <= m_start or td_start >= m_end)
            for td_start, td_end in time_date_spans
        ):
            continue
        raw = match.group()
        has_currency = raw[0] in CURRENCY_SYMBOLS
        cleaned = raw.replace(",", "")
        if has_currency:
            cleaned = cleaned[1:]
        claim_type = "currency" if has_currency else "numeric"
        regex_claims.append(
            Claim(
                claim_id=f"claim_{claim_index}",
                claim_type=claim_type,
                text_span=TextSpan(
                    start=match.start(),
                    end=match.end(),
                    text=raw,
                ),
                extracted_value=cleaned,
                source_ref=None,
            )
        )
        claim_index += 1

    nlp = _load_spacy_model()
    doc = nlp(agent_output)

    spacy_claims: list[Claim] = []
    for ent in doc.ents:
        if ent.label_ in {"ORG", "GPE", "PERSON", "PRODUCT", "FAC", "LOC"}:
            claim_type = "string"
        elif ent.label_ == "MONEY":
            claim_type = "currency"
        elif ent.label_ == "DATE":
            claim_type = "date"
        else:
            continue

        spacy_claims.append(
            Claim(
                claim_id="",
                claim_type=claim_type,
                text_span=TextSpan(
                    start=ent.start_char,
                    end=ent.end_char,
                    text=ent.text,
                ),
                extracted_value=ent.text,
                source_ref=None,
            )
        )

    spacy_spans = [
        (claim.text_span.start, claim.text_span.end) for claim in spacy_claims
    ]

    filtered_regex_claims: list[Claim] = []
    for claim in regex_claims:
        r_start = claim.text_span.start
        r_end = claim.text_span.end
        overlaps = any(not (s_end <= r_start or s_start >= r_end) for s_start, s_end in spacy_spans)
        if not overlaps:
            filtered_regex_claims.append(claim)

    merged = spacy_claims + filtered_regex_claims
    merged.sort(key=lambda c: c.text_span.start)

    for i, claim in enumerate(merged):
        claim.claim_id = f"claim_{i}"

    return merged
