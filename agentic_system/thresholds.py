"""
Confidence thresholds for the multi-agent system.

These gates control when low self-assessment or low critic confidence
forces revision / blocks auto-approve.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceThresholds:
    """
    Tunable gates used by QA agents and the QA-Orchestrator.

    Domain (self-assessment) gates
    --------------------------------
    domain_low: below this → QA emits a major "completeness" issue
    domain_critical: below this → QA emits a critical issue
    domain_approve_min: orchestrator will not auto-approve if any
        selected domain agent is below this

    QA (critic) gates
    -----------------
    qa_approve_min: average critic confidence must be >= this to approve
        (when there are no major/critical issues)

    Orchestrator policy
    -------------------
    quality_approve_min: QualityDecision.quality_score floor for approve
        (already enforced in the Pydantic model at 0.5; this is the
         operational target used when building the score)
    """

    domain_low: float = 0.70
    domain_critical: float = 0.50
    domain_approve_min: float = 0.75

    qa_approve_min: float = 0.80

    quality_approve_min: float = 0.85

    def __post_init__(self) -> None:
        for name in (
            "domain_low",
            "domain_critical",
            "domain_approve_min",
            "qa_approve_min",
            "quality_approve_min",
        ):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        if self.domain_critical > self.domain_low:
            raise ValueError("domain_critical must be <= domain_low")
        if self.domain_low > self.domain_approve_min:
            raise ValueError("domain_low should be <= domain_approve_min")


# Default production-ish gates
DEFAULT_THRESHOLDS = ConfidenceThresholds()
"""
DEFAULT_THRESHOLDS = ConfidenceThresholds()
"""
