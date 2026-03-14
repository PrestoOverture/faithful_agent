"""Shared Pydantic models: ToolCall, Claim, MatchResult, Report."""

from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, model_validator


class TextSpan(BaseModel):
    start: int
    end: int
    text: str

    @model_validator(mode="after")
    def end_gte_start(self) -> "TextSpan":
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")
        return self


class SourceRef(BaseModel):
    call_id: str
    path: str
    transform: str = "none"


class ToolCall(BaseModel):
    call_id: str
    tool_name: str
    request: dict = {}
    response: dict
    timestamp: Optional[datetime] = None


class Claim(BaseModel):
    claim_id: str
    claim_type: Literal["numeric", "string", "date", "currency"]
    text_span: TextSpan
    extracted_value: Any
    source_ref: Optional[SourceRef] = None


class MatchResult(BaseModel):
    claim_id: str
    matched_call_id: Optional[str] = None
    matched_field: Optional[str] = None
    expected_value: Any = None
    observed_value: Any = None
    status: Literal["pass", "warning", "error"]
    mismatch_category: Optional[
        Literal[
            "exact_mismatch",
            "rounding",
            "unit_conversion",
            "derived",
            "omission",
            "unsupported",
        ]
    ] = None
    severity: Literal["info", "warning", "error"]
    match_confidence: Literal["high", "medium", "low"] = "high"
    diff: Optional[dict] = None
    recommended_action: Optional[str] = None


class ReportSummary(BaseModel):
    total_claims: int
    passed: int
    warnings: int
    errors: int
    verdict: Literal["pass", "warn", "fail"]


class Report(BaseModel):
    verification_id: str = ""
    timestamp: Optional[datetime] = None
    mode: Literal["strict", "heuristic"]
    summary: ReportSummary
    claims: list[MatchResult]

    def model_post_init(self, __context: Any) -> None:
        if not self.verification_id:
            object.__setattr__(self, "verification_id", str(uuid4()))
        if self.timestamp is None:
            object.__setattr__(self, "timestamp", datetime.now(timezone.utc))
