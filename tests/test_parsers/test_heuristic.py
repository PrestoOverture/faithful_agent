"""Tests for faithfulagent/parsers/heuristic.py — spaCy NER path coverage."""

import warnings
from unittest.mock import patch

import pytest

from faithfulagent.parsers.heuristic import parse_heuristic
from faithfulagent.schemas import ToolCall

_TC_EMPTY: list[ToolCall] = []


# ---------------------------------------------------------------------------
# Regex-only fallback when spaCy is not available
# ---------------------------------------------------------------------------


def test_regex_fallback_when_spacy_missing():
    """parse_heuristic returns regex claims and does not raise when spaCy is absent."""
    with patch(
        "faithfulagent.parsers.heuristic._load_spacy_model",
        side_effect=ImportError("spaCy not installed"),
    ):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            claims = parse_heuristic("The price is 553 dollars", _TC_EMPTY)

    assert len(claims) == 1
    assert claims[0].claim_type == "numeric"
    assert claims[0].extracted_value == "553"
    assert claims[0].claim_id == "claim_0"


def test_regex_fallback_issues_warning():
    """A UserWarning is emitted when falling back to regex-only mode."""
    with patch(
        "faithfulagent.parsers.heuristic._load_spacy_model",
        side_effect=ImportError("spaCy not installed"),
    ):
        with pytest.warns(UserWarning, match="spaCy not available"):
            parse_heuristic("The price is 553 dollars", _TC_EMPTY)


def test_regex_fallback_returns_multiple_claims():
    """All regex-matched numbers are returned in fallback mode."""
    with patch(
        "faithfulagent.parsers.heuristic._load_spacy_model",
        side_effect=ImportError("spaCy not installed"),
    ):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            claims = parse_heuristic("costs 553 and takes 270 minutes", _TC_EMPTY)

    values = [c.extracted_value for c in claims]
    assert "553" in values
    assert "270" in values
    assert [c.claim_id for c in claims] == [f"claim_{i}" for i in range(len(claims))]


spacy = pytest.importorskip("spacy", reason="spaCy not installed")

# Also skip if the model is not downloaded.
try:
    spacy.load("en_core_web_sm")
except OSError:
    pytest.skip("en_core_web_sm model not installed", allow_module_level=True)

from faithfulagent.parsers.heuristic import parse_heuristic
from faithfulagent.schemas import ToolCall

# Minimal tool_calls — heuristic parser does not use them, but the signature requires them.
_TC: list[ToolCall] = [ToolCall(call_id="c1", tool_name="t", response={"x": 1})]


# ---------------------------------------------------------------------------
# Entity type mapping
# ---------------------------------------------------------------------------


def test_org_entity_maps_to_string():
    claims = parse_heuristic("Apple is a great company.", _TC)
    assert len(claims) == 1
    c = claims[0]
    assert c.claim_type == "string"
    assert c.extracted_value == "Apple"
    assert c.text_span.start == 0
    assert c.text_span.end == 5


def test_gpe_entity_maps_to_string():
    claims = parse_heuristic("The team is based in London.", _TC)
    assert any(c.claim_type == "string" and c.extracted_value == "London" for c in claims)


def test_person_entity_maps_to_string():
    claims = parse_heuristic("John Smith signed the contract.", _TC)
    assert any(c.claim_type == "string" and c.extracted_value == "John Smith" for c in claims)


def test_loc_entity_maps_to_string():
    # "the Pacific Ocean" is tagged LOC by en_core_web_sm
    claims = parse_heuristic("The team flew over the Pacific Ocean.", _TC)
    assert any(c.claim_type == "string" for c in claims)
    loc_claims = [c for c in claims if c.claim_type == "string"]
    assert loc_claims[0].extracted_value == "the Pacific Ocean"


def test_fac_entity_maps_to_string():
    # "Rockefeller Center" is tagged FAC by en_core_web_sm
    claims = parse_heuristic("The building at Rockefeller Center was huge.", _TC)
    assert any(c.claim_type == "string" and c.extracted_value == "Rockefeller Center" for c in claims)


def test_money_entity_maps_to_currency():
    # spaCy tags "553" as MONEY (the $ prefix is captured by regex but spaCy wins)
    claims = parse_heuristic("The price is $553.", _TC)
    assert len(claims) == 1
    c = claims[0]
    assert c.claim_type == "currency"
    assert c.extracted_value == "553"  # spaCy entity text (without $)


def test_date_entity_maps_to_date():
    claims = parse_heuristic("The event is on January 15th.", _TC)
    assert any(c.claim_type == "date" and c.extracted_value == "January 15th" for c in claims)


# ---------------------------------------------------------------------------
# Overlap deduplication: spaCy entity wins over regex match
# ---------------------------------------------------------------------------


def test_spacy_money_wins_over_regex_currency():
    # Regex produces "$553" (currency); spaCy produces "553" (MONEY) — they overlap.
    # Only the spaCy entity should survive.
    text = "The flight costs $553."
    claims = parse_heuristic(text, _TC)
    # Exactly one claim — the spaCy MONEY entity
    assert len(claims) == 1
    assert claims[0].claim_type == "currency"
    # extracted_value is the spaCy entity text, not the stripped regex value
    assert claims[0].extracted_value == "553"
    # Span corresponds to spaCy entity (chars 18–21, just the digits)
    assert claims[0].text_span.start == 18
    assert claims[0].text_span.end == 21


def test_no_regex_duplicate_when_number_is_date_entity():
    # "1976" is tagged DATE by spaCy; regex would also catch it as numeric.
    # Only the spaCy DATE entity should survive.
    text = "Apple was founded in 1976."
    claims = parse_heuristic(text, _TC)
    # There should be no numeric claim for "1976" — only DATE and ORG
    numeric_claims = [c for c in claims if c.claim_type == "numeric"]
    # No plain-numeric claim for the year (it's covered by spaCy DATE)
    assert all(c.extracted_value != "1976" for c in numeric_claims)
    date_claims = [c for c in claims if c.claim_type == "date"]
    assert any(c.extracted_value == "1976" for c in date_claims)


# ---------------------------------------------------------------------------
# Mixed extraction: regex-only numbers + spaCy entities coexist
# ---------------------------------------------------------------------------


def test_mixed_regex_and_spacy_no_overlap():
    # "Apple" → ORG (string); "500" → spaCy tags as CARDINAL (ignored by our mapping),
    # so regex keeps it as numeric.
    text = "Apple earned 500 million."
    claims = parse_heuristic(text, _TC)
    types = {c.claim_type for c in claims}
    assert "string" in types   # Apple (ORG)
    assert "numeric" in types  # 500 (regex, CARDINAL is ignored by spaCy path)
    assert len(claims) == 2


def test_multi_entity_output_ordering():
    # Input has MONEY ($553 at ~17), GPE (London at ~39), DATE (March 10 at ~49).
    text = "The flight costs $553 and departs from London on March 10."
    claims = parse_heuristic(text, _TC)
    assert len(claims) == 3
    # Claims must be ordered by span start
    starts = [c.text_span.start for c in claims]
    assert starts == sorted(starts)
    # Claim IDs must be assigned in order
    assert [c.claim_id for c in claims] == ["claim_0", "claim_1", "claim_2"]
    # Types in span order
    assert claims[0].claim_type == "currency"  # $553 / MONEY
    assert claims[1].claim_type == "string"    # London / GPE
    assert claims[2].claim_type == "date"      # March 10 / DATE


# ---------------------------------------------------------------------------
# Claim ID reassignment after merge
# ---------------------------------------------------------------------------


def test_claim_ids_reassigned_after_merge():
    # Confirm IDs are sequential from 0 regardless of how many entities spaCy found.
    text = "John Smith works at Apple in London."
    claims = parse_heuristic(text, _TC)
    assert [c.claim_id for c in claims] == [f"claim_{i}" for i in range(len(claims))]


# ---------------------------------------------------------------------------
# Non-mapped entity types are excluded
# ---------------------------------------------------------------------------


def test_cardinal_entity_not_captured_by_spacy_path():
    # spaCy tags "5" as CARDINAL — not in our entity map.
    # The number should still appear via regex as "numeric", not via spaCy.
    text = "There were 5 apples."
    claims = parse_heuristic(text, _TC)
    assert len(claims) == 1
    assert claims[0].claim_type == "numeric"
    assert claims[0].extracted_value == "5"


# ---------------------------------------------------------------------------
# source_ref is always None in heuristic mode
# ---------------------------------------------------------------------------


def test_spacy_claims_have_no_source_ref():
    claims = parse_heuristic("Apple is great.", _TC)
    for c in claims:
        assert c.source_ref is None


# ---------------------------------------------------------------------------
# Time/date pattern exclusion: digits inside time/date spans are not claims
# ---------------------------------------------------------------------------


def test_time_only_text_yields_no_numeric_claims():
    # All digits are inside time patterns — nothing should survive
    claims = parse_heuristic("departs at 08:30 AM and arrives at 01:00 PM", _TC)
    numeric_claims = [c for c in claims if c.claim_type == "numeric"]
    assert len(numeric_claims) == 0


def test_number_before_time_not_suppressed():
    # 553 is outside any time pattern; 8:30 AM digits should be suppressed
    claims = parse_heuristic("costs 553 and departs at 8:30 AM", _TC)
    numeric_claims = [c for c in claims if c.claim_type == "numeric"]
    values = [c.extracted_value for c in numeric_claims]
    assert "553" in values
    assert "8" not in values
    assert "30" not in values


def test_iso_datetime_yields_no_numeric_claims():
    claims = parse_heuristic("date is 2025-08-15T08:30:00", _TC)
    numeric_claims = [c for c in claims if c.claim_type == "numeric"]
    assert len(numeric_claims) == 0


def test_numbers_around_time_only_non_time_extracted():
    # 270 and 2 are outside time patterns; 3 and 45 are inside "3:45 PM"
    claims = parse_heuristic("price is 270 at 3:45 PM with 2 stops", _TC)
    numeric_claims = [c for c in claims if c.claim_type == "numeric"]
    values = [c.extracted_value for c in numeric_claims]
    assert "270" in values
    assert "2" in values
    assert "3" not in values
    assert "45" not in values
