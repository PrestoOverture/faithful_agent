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


class Constraint(BaseModel):
    constraint_id: str
    constraint_type: Literal["comparison", "ranking", "sum"]
    lhs_claim_id: Optional[str] = None
    operator: Optional[Literal["<", ">", "<=", ">=", "=="]] = None
    rhs_claim_id: Optional[str] = None
    ranked_claim_id: Optional[str] = None
    rank_type: Optional[Literal["min", "max"]] = None
    candidate_claim_ids: Optional[list[str]] = None
    total_claim_id: Optional[str] = None
    component_claim_ids: Optional[list[str]] = None

    @model_validator(mode="after")
    def check_fields_for_type(self) -> "Constraint":
        if self.constraint_type == "comparison":
            if not (self.lhs_claim_id and self.operator and self.rhs_claim_id):
                raise ValueError(
                    "comparison constraints require lhs_claim_id, operator, rhs_claim_id"
                )
        elif self.constraint_type == "ranking":
            if not (
                self.ranked_claim_id
                and self.rank_type
                and self.candidate_claim_ids is not None
            ):
                raise ValueError(
                    "ranking constraints require ranked_claim_id, rank_type, candidate_claim_ids"
                )
            if self.ranked_claim_id not in self.candidate_claim_ids:
                raise ValueError("ranked_claim_id must be in candidate_claim_ids")
            if len(self.candidate_claim_ids) < 2:
                raise ValueError("ranking constraints require at least 2 candidates")
        elif self.constraint_type == "sum":
            if not (self.total_claim_id and self.component_claim_ids is not None):
                raise ValueError(
                    "sum constraints require total_claim_id and component_claim_ids"
                )
            if len(self.component_claim_ids) < 2:
                raise ValueError("sum constraints require at least 2 components")
        return self


class CrossClaimResult(BaseModel):
    constraint_id: str
    constraint_type: Literal["comparison", "ranking", "sum"]
    status: Literal["pass", "error"]
    violation_category: Optional[
        Literal["comparison_violated", "ranking_violated", "sum_mismatch"]
    ] = None
    severity: Literal["error", "warning"] = "error"
    match_confidence: Literal["high", "medium"] = "high"
    involved_claim_ids: list[str]
    observed_values: dict[str, Any]
    expected_relationship: str
    diff: Optional[dict] = None
    recommended_action: Optional[str] = None


class ReportSummary(BaseModel):
    total_claims: int
    passed: int
    warnings: int
    errors: int
    cross_claim_violations: int = 0
    verdict: Literal["pass", "warn", "fail"]


class Report(BaseModel):
    verification_id: str = ""
    timestamp: Optional[datetime] = None
    mode: Literal["strict", "heuristic"]
    summary: ReportSummary
    claims: list[MatchResult]
    cross_claim_results: list[CrossClaimResult] = []

    def model_post_init(self, __context: Any) -> None:
        if not self.verification_id:
            object.__setattr__(self, "verification_id", str(uuid4()))
        if self.timestamp is None:
            object.__setattr__(self, "timestamp", datetime.now(timezone.utc))
