"""
State schema for the multi-agent system.
Designed for: Main Orchestrator + N Domain Agents + N QA Agents + QA-Orchestrator

All structured models include explicit validation so LLM outputs
cannot silently produce invalid state.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, TypedDict
from typing_extensions import NotRequired
import operator

from pydantic import BaseModel, Field, field_validator, model_validator
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Nested models (with structured output validation)
# ---------------------------------------------------------------------------

class DomainOutput(BaseModel):
    agent_id: str = Field(min_length=1)
    content: str | dict
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    confidence_justification: str = Field(
        default="",
        description="Agent's own explanation of why it assigned this confidence.",
    )
    tools_used: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    @field_validator("agent_id")
    @classmethod
    def agent_id_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("agent_id cannot be blank")
        return v

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str | dict):
        if isinstance(v, str) and not v.strip():
            raise ValueError("content cannot be an empty string")
        if isinstance(v, dict) and not v:
            raise ValueError("content dict cannot be empty")
        return v

    @field_validator("confidence_justification")
    @classmethod
    def justification_stripped(cls, v: str) -> str:
        return (v or "").strip()


class QAIssue(BaseModel):
    severity: Literal["critical", "major", "minor"]
    category: Literal[
        "factual", "safety", "completeness", "style", "consistency", "other"
    ] = "other"
    description: str = Field(min_length=3)
    evidence: Optional[str] = None
    suggested_fix: Optional[str] = None

    @field_validator("description")
    @classmethod
    def description_not_blank(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("description must be at least 3 characters")
        return v


class QAReport(BaseModel):
    agent_id: str = Field(min_length=1)
    verdict: Literal["pass", "fail", "conditional"]
    severity: Literal["critical", "major", "minor", "none"] = "none"
    confidence: float = Field(ge=0.0, le=1.0, default=0.9)
    issues: list[QAIssue] = Field(default_factory=list)
    summary: str = ""
    raw_critique: Optional[str] = None

    @field_validator("agent_id")
    @classmethod
    def agent_id_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("agent_id cannot be blank")
        return v

    @model_validator(mode="after")
    def verdict_consistency(self) -> "QAReport":
        """Enforce consistency between verdict, severity, and issues."""
        if self.verdict == "pass":
            if self.issues and any(i.severity in ("critical", "major") for i in self.issues):
                raise ValueError(
                    "verdict='pass' is incompatible with critical/major issues"
                )
            if self.severity in ("critical", "major"):
                raise ValueError(
                    "verdict='pass' is incompatible with severity='critical' or 'major'"
                )
        if self.verdict == "fail" and not self.issues and self.severity == "none":
            # Soft warning path — force at least a severity
            object.__setattr__(self, "severity", "major")
        return self


class QualityDecision(BaseModel):
    disposition: Literal["approve", "revise", "escalate", "reject"]
    rationale: str = Field(min_length=5)
    quality_score: float = Field(ge=0.0, le=1.0)
    cross_agent_conflicts: list[str] = Field(default_factory=list)
    required_actions: list[dict] = Field(default_factory=list)
    revision_count_at_decision: int = Field(ge=0, default=0)

    @field_validator("rationale")
    @classmethod
    def rationale_not_blank(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 5:
            raise ValueError("rationale must be at least 5 characters")
        return v

    @field_validator("required_actions")
    @classmethod
    def actions_must_have_target(cls, v: list[dict]) -> list[dict]:
        cleaned = []
        for action in v:
            if not isinstance(action, dict):
                continue
            if "target" not in action or not str(action["target"]).strip():
                continue  # drop malformed actions
            cleaned.append(action)
        return cleaned

    @model_validator(mode="after")
    def disposition_consistency(self) -> "QualityDecision":
        if self.disposition == "approve" and self.quality_score < 0.5:
            raise ValueError(
                "disposition='approve' requires quality_score >= 0.5"
            )
        if self.disposition == "revise" and not self.required_actions:
            # Allow empty actions but it's a soft smell; we don't hard-fail
            pass
        if self.disposition in ("reject", "escalate") and self.quality_score > 0.85:
            raise ValueError(
                f"disposition='{self.disposition}' is inconsistent with high quality_score"
            )
        return self


# ---------------------------------------------------------------------------
# Main graph state
# ---------------------------------------------------------------------------

class AgentSystemState(TypedDict):
    # Conversation / task identity
    messages: Annotated[list, add_messages]
    original_goal: str
    task_id: str

    # Domain layer (parallel-safe)
    domain_outputs: Annotated[dict[str, DomainOutput], operator.or_]

    # QA layer (parallel-safe)
    qa_reports: Annotated[list[QAReport], operator.add]

    # Quality control layer (written only by QA-Orchestrator)
    quality_decision: NotRequired[Optional[QualityDecision]]

    # Control plane
    revision_count: int
    max_revisions: int
    current_phase: Literal[
        "planning",
        "domain_execution",
        "qa_execution",
        "qa_orchestration",
        "revision",
        "finalize",
        "escalated",
        "rejected",
    ]
    next_action: NotRequired[Optional[str]]

    # Selective routing — which domain agents the orchestrator decided to run
    selected_agents: NotRequired[list[str]]

    # Observability
    audit_log: Annotated[list[dict], operator.add]
    error: NotRequired[Optional[str]]
    human_feedback: NotRequired[Optional[str]]
