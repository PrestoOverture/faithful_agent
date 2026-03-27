"""Strict mode parser: extracts claims from structured agent annotations."""

from typing import Any

from jsonpath_ng import parse as jp_parse

from faithfulagent.schemas import Claim, Constraint, SourceRef, TextSpan, ToolCall


def parse_strict(
    agent_output: dict, tool_calls: list[ToolCall]
) -> tuple[list[Claim], list[Constraint]]:
    text = agent_output["text"]
    refs = agent_output.get("refs", [])
    claims: list[Claim] = []
    constraints: list[Constraint] = []

    tool_call_map = {call.call_id: call for call in tool_calls}

    for i, ref in enumerate(refs):
        call_id = ref["call_id"]
        tool_call = tool_call_map.get(call_id)
        if tool_call is None:
            raise ValueError(f"No tool call found with call_id '{call_id}'")

        span = ref["span"]
        span_text = text[span[0] : span[1]]

        matches = jp_parse(ref["path"]).find(tool_call.response)
        resolved: Any = matches[0].value if matches else None

        if isinstance(resolved, (int, float)):
            claim_type = "numeric"
        elif isinstance(resolved, str):
            claim_type = "string"
        else:
            claim_type = "string"

        claims.append(
            Claim(
                claim_id=f"claim_{i}",
                claim_type=claim_type,
                text_span=TextSpan(start=span[0], end=span[1], text=span_text),
                extracted_value=span_text,
                source_ref=SourceRef(
                    call_id=call_id,
                    path=ref["path"],
                    transform=ref.get("transform", "none"),
                ),
            )
        )

    seen_constraint_ids: set[str] = set()

    def get_numeric_claim(ref_index: int) -> Claim:
        if ref_index < 0 or ref_index >= len(refs):
            raise ValueError(f"Constraint ref_index out of bounds: {ref_index}")

        claim = claims[ref_index]
        if claim.claim_type not in ("numeric", "currency"):
            raise ValueError(
                f"Constraint references non-numeric claim at ref_index {ref_index}"
            )
        return claim

    for raw_constraint in agent_output.get("constraints", []):
        constraint_id = raw_constraint["constraint_id"]
        if constraint_id in seen_constraint_ids:
            raise ValueError(f"Duplicate constraint_id: {constraint_id}")
        seen_constraint_ids.add(constraint_id)

        constraint_type = raw_constraint["constraint_type"]

        if constraint_type == "comparison":
            lhs_claim = get_numeric_claim(raw_constraint["lhs_ref_index"])
            rhs_claim = get_numeric_claim(raw_constraint["rhs_ref_index"])
            constraints.append(
                Constraint(
                    constraint_id=constraint_id,
                    constraint_type=constraint_type,
                    lhs_claim_id=lhs_claim.claim_id,
                    operator=raw_constraint["operator"],
                    rhs_claim_id=rhs_claim.claim_id,
                )
            )
        elif constraint_type == "ranking":
            ranked_claim = get_numeric_claim(raw_constraint["ranked_ref_index"])
            candidate_claim_ids = [
                get_numeric_claim(ref_index).claim_id
                for ref_index in raw_constraint["candidate_ref_indices"]
            ]
            constraints.append(
                Constraint(
                    constraint_id=constraint_id,
                    constraint_type=constraint_type,
                    ranked_claim_id=ranked_claim.claim_id,
                    rank_type=raw_constraint["rank_type"],
                    candidate_claim_ids=candidate_claim_ids,
                )
            )
        elif constraint_type == "sum":
            total_claim = get_numeric_claim(raw_constraint["total_ref_index"])
            component_claim_ids = [
                get_numeric_claim(ref_index).claim_id
                for ref_index in raw_constraint["component_ref_indices"]
            ]
            constraints.append(
                Constraint(
                    constraint_id=constraint_id,
                    constraint_type=constraint_type,
                    total_claim_id=total_claim.claim_id,
                    component_claim_ids=component_claim_ids,
                )
            )

    return claims, constraints
