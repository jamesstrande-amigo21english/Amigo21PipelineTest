"""
Agent implementations for the multi-agent system.

In a real deployment these would call LLMs (OpenAI, Anthropic, Grok, etc.).
Here we use deterministic mock logic so the full graph is runnable without API keys.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError

from .state import (
    AgentSystemState,
    DomainOutput,
    QAIssue,
    QAReport,
    QualityDecision,
)
from .logging_config import get_logger

logger = get_logger("agents")


# ---------------------------------------------------------------------------
# Domain Agents (specialists) — Draft + Refine
# ---------------------------------------------------------------------------

def _extract_feedback_for_agent(state: AgentSystemState, agent_id: str) -> str:
    """Pull targeted QA / orchestrator feedback for this agent."""
    parts: list[str] = []

    decision = state.get("quality_decision")
    if decision and decision.required_actions:
        for action in decision.required_actions:
            if action.get("target") == agent_id:
                parts.append(action.get("instruction", ""))

    # Also surface issues from the most recent QA report for this agent
    for report in reversed(state.get("qa_reports", [])):
        if report.agent_id == agent_id and report.issues:
            for issue in report.issues:
                line = f"[{issue.severity}] {issue.category}: {issue.description}"
                if issue.suggested_fix:
                    line += f" → Fix: {issue.suggested_fix}"
                parts.append(line)
            break

    return "\n".join(p for p in parts if p).strip()


# Structured self-assessment schema returned by domain agents
class DomainAgentResponse(BaseModel):
    """
    Structured output from a domain agent (draft or refine).
    Forces explicit self-assessment instead of silent heuristic confidence.
    """
    content: str = Field(min_length=1, description="Full specialist output.")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Your calibrated confidence in this output (0–1). "
            "Use lower values when evidence is thin, claims are speculative, "
            "or important angles are missing."
        ),
    )
    confidence_justification: str = Field(
        min_length=10,
        description=(
            "Brief explanation of the confidence score: what supports it, "
            "what remains uncertain, and any residual risks."
        ),
    )
    residual_risks: list[str] = Field(
        default_factory=list,
        description="Specific residual risks or open questions (if any).",
    )

    @field_validator("confidence_justification")
    @classmethod
    def justification_not_blank(cls, v: str) -> str:
        v = (v or "").strip()
        if len(v) < 10:
            raise ValueError("confidence_justification must be at least 10 characters")
        return v


def make_domain_agent(agent_id: str, specialty: str, model=None):
    """
    Factory for a domain specialist with structured self-assessment.

    Behavior:
      - revision == 0  →  DRAFT  (fresh generation)
      - revision  > 0  →  REFINE (previous output + QA critique → improved output)

    Both paths require the agent to report confidence + justification.
    """

    def domain_node(state: AgentSystemState) -> dict[str, Any]:
        goal = state["original_goal"]
        revision = state.get("revision_count", 0)
        task_id = state.get("task_id", "?")
        mode = "refine" if revision > 0 else "draft"

        logger.info(
            "Domain agent starting | agent=%s mode=%s specialty=%s revision=%s task=%s",
            agent_id, mode, specialty, revision, task_id,
        )

        try:
            previous = state.get("domain_outputs", {}).get(agent_id)
            feedback = _extract_feedback_for_agent(state, agent_id)

            if model is not None:
                assessment = _llm_domain_call(
                    model=model,
                    agent_id=agent_id,
                    specialty=specialty,
                    goal=goal,
                    mode=mode,
                    previous_content=previous.content if previous else None,
                    feedback=feedback,
                    revision=revision,
                )
            else:
                assessment = _mock_domain_call(
                    agent_id=agent_id,
                    specialty=specialty,
                    goal=goal,
                    mode=mode,
                    previous=previous,
                    feedback=feedback,
                    revision=revision,
                )

            output = DomainOutput(
                agent_id=agent_id,
                content=assessment.content,
                confidence=assessment.confidence,
                confidence_justification=assessment.confidence_justification,
                tools_used=["llm_draft"] if mode == "draft" else ["llm_refine", "qa_feedback"],
                metadata={
                    "specialty": specialty,
                    "revision": revision,
                    "mode": mode,
                    "had_feedback": bool(feedback),
                    "residual_risks": assessment.residual_risks,
                },
            )

            logger.info(
                "Domain agent completed | agent=%s mode=%s confidence=%.2f justification=%s",
                agent_id, mode, assessment.confidence,
                assessment.confidence_justification[:60],
            )

            return {
                "domain_outputs": {agent_id: output},
                "audit_log": [
                    {
                        "event": "domain_agent_completed",
                        "agent_id": agent_id,
                        "mode": mode,
                        "confidence": assessment.confidence,
                        "confidence_justification": assessment.confidence_justification,
                        "residual_risks": assessment.residual_risks,
                        "had_feedback": bool(feedback),
                    }
                ],
            }
        except Exception as e:
            logger.exception(
                "Domain agent failed | agent=%s mode=%s task=%s error=%s",
                agent_id, mode, task_id, e,
            )
            return {
                "audit_log": [
                    {
                        "event": "domain_agent_error",
                        "agent_id": agent_id,
                        "mode": mode,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                ],
                "error": f"domain_agent:{agent_id}:{e}",
            }

    domain_node.__name__ = f"domain_{agent_id}"
    return domain_node


def _mock_page_content(agent_id: str, specialty: str, goal: str) -> str:
    """Deterministic Amigo 21 page drafts for MOCK mode (no live calls)."""
    pages = {
        "home": (
            "# Home — Amigo 21 English\n"
            "Hero: English for Portuguese-speaking learners (EN + PT-BR).\n"
            "Value: twelve courses across three tracks; placement-aware enrollment.\n"
            "CTAs: Explore Courses | See Pricing (informational) | Contact Us.\n"
            "Canonical domain target: amigo21english.com (marketing site pipeline).\n"
        ),
        "courses": (
            "# Courses / Tracks — Amigo 21 English\n"
            "Track 1: Beginner I/II, Elementary I/II, Intermediate I/II, Advanced I/II.\n"
            "Track 2: Professional English I/II.\n"
            "Track 3: Driver's Exam English I/II (not affiliated with NJ MVC).\n"
            "Format: eight-week courses, 50-minute lessons.\n"
            "Placement: only 'Track 1 — Course 1: Beginner I' is exempt; all other "
            "valid selections open the teacher-guided placement experience.\n"
        ),
        "about": (
            "# About — Amigo 21 English\n"
            "Mission: bilingual public website for Portuguese-speaking learners "
            "seeking practical English instruction.\n"
            "Offering: business contact links, course summaries, inquiry and "
            "placement entrypoints. Public pages summarize teaching resources; "
            "they do not publish private lesson plans or teacher answer keys.\n"
        ),
        "pricing": (
            "# Pricing — Amigo 21 English (NONPRODUCTION / informational only)\n"
            "WARNING: This page shows informational tuition only. It is NOT a live "
            "checkout catalog. Payments remain strictly nonproduction.\n"
            "Matrix (USD Basic / Complete / Plus):\n"
            "- Beginner I/II: 149 / 199 / 249\n"
            "- Elementary I/II: 210 / 249 / 299\n"
            "- Intermediate I/II: 249 / 279 / 329\n"
            "- Advanced I/II: 299 / 329 / 399\n"
            "- Professional English I/II: 349 / 419 / 499\n"
            "- Driver's Exam English I/II: 499 / 529 / 599\n"
            "Do not initiate payment provider calls from this pipeline mock.\n"
        ),
        "contact": (
            "# Contact — Amigo 21 English\n"
            "Inquiry form fields: firstName, lastName, email, phone (optional), "
            "lessonInterest, message, hidden website honeypot.\n"
            "Required: firstName, lastName, email, valid course interest, message.\n"
            "Frontend contract: POST to existing production backend "
            "https://amigo21website-7doj6ahycq-ue.a.run.app/api/contact "
            "(also reachable as relative /api/contact on the Node app).\n"
            "MOCK MODE: document the contract only — do NOT perform a real POST.\n"
            "Health: GET; CORS: OPTIONS. Honeypot returns success-like without insert.\n"
        ),
        "faq": (
            "# FAQ — Amigo 21 English\n"
            "Q: How many courses? A: Twelve courses across three tracks.\n"
            "Q: Lesson length? A: 50-minute lessons over eight weeks.\n"
            "Q: Placement test? A: Required except Beginner I exemption.\n"
            "Q: Languages? A: English and Brazilian Portuguese experiences.\n"
            "Q: Payments live? A: Public marketing site is informational; "
            "payments stay in nonproduction sandboxes.\n"
            "Q: Contact? A: Use the Contact form (POST /api/contact backend).\n"
        ),
    }
    body = pages.get(
        agent_id,
        f"Page draft for '{agent_id}' covering specialty: {specialty}.\n",
    )
    return (
        f"[{agent_id.upper()} | {specialty}] PAGE DRAFT\n"
        f"Goal: {goal}\n\n"
        f"{body}"
    )


def _mock_domain_call(
    agent_id: str,
    specialty: str,
    goal: str,
    mode: str,
    previous: DomainOutput | None,
    feedback: str,
    revision: int,
) -> DomainAgentResponse:
    """Deterministic self-assessment for demo mode."""
    if mode == "draft":
        return DomainAgentResponse(
            content=_mock_page_content(agent_id, specialty, goal),
            confidence=0.78,
            confidence_justification=(
                "Solid initial coverage of the main dimensions, but evidence is "
                "still thin and cross-agent consistency has not been checked."
            ),
            residual_risks=[
                "Limited primary sources",
                "Possible inconsistency with peer agents",
            ],
        )

    # Refine path — confidence rises because feedback was addressed
    prev_conf = previous.confidence if previous else 0.7
    new_conf = min(0.95, prev_conf + 0.12)
    base = previous.content if previous else _mock_page_content(agent_id, specialty, goal)
    return DomainAgentResponse(
        content=(
            f"[{agent_id.upper()} | {specialty}] REFINED (cycle {revision})\n"
            f"Goal: {goal}\n"
            f"Previous draft was revised using QA feedback.\n"
            f"Feedback addressed:\n{feedback or '(none)'}\n\n"
            f"Improved page content:\n{base}"
        ),
        confidence=new_conf,
        confidence_justification=(
            f"Revised against explicit QA feedback. Prior confidence was {prev_conf:.2f}; "
            f"key consistency and completeness issues were addressed, raising confidence "
            f"to {new_conf:.2f}."
        ),
        residual_risks=["Minor gaps may remain after a single refine pass"],
    )


def _llm_domain_call(
    model,
    agent_id: str,
    specialty: str,
    goal: str,
    mode: str,
    previous_content: str | dict | None,
    feedback: str,
    revision: int,
) -> DomainAgentResponse:
    """
    Real LLM draft/refine with structured self-assessment.
    Returns a validated DomainAgentResponse (content + confidence + justification).
    """
    self_assess_rules = """
You MUST perform structured self-assessment:
- confidence: calibrated 0–1. Do NOT default to high values.
  0.9+ only if evidence is strong and residual risk is low.
  0.6–0.8 if the analysis is useful but incomplete or uncertain.
  <0.6 if major gaps or speculation remain.
- confidence_justification: explain the score in concrete terms.
- residual_risks: list specific open risks (empty only if truly none).
"""

    if mode == "draft":
        system = f"""You are a specialist agent: {specialty}.
Produce a high-quality first draft for the given goal.
Be precise, structured, and evidence-oriented.
This is a DRAFT — focus on coverage and clarity; critique will follow.
{self_assess_rules}"""
        human = f"Goal:\n{goal}"
    else:
        system = f"""You are a specialist agent: {specialty}.
You previously produced a draft. A QA critic has provided feedback.
REFINE the draft: address every feedback point, keep what was strong,
and produce a full improved version (not a patch note).
{self_assess_rules}"""
        human = (
            f"Goal:\n{goal}\n\n"
            f"Previous draft:\n{previous_content}\n\n"
            f"QA feedback to address:\n{feedback or '(no specific feedback)'}\n\n"
            f"Revision cycle: {revision}"
        )

    structured = model.with_structured_output(DomainAgentResponse)
    result: DomainAgentResponse = structured.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": human},
    ])
    return result


# ---------------------------------------------------------------------------
# Per-agent QA Agents
# ---------------------------------------------------------------------------

def make_qa_agent(agent_id: str, model=None, thresholds=None):
    """
    Factory for a dedicated QA critic agent.

    This is the CRITIQUE step in the draft → critique → refine loop.
    When `model` is provided, uses a real LLM with structured QAReport output.
    Otherwise uses deterministic mock logic (demo mode).

    `thresholds` gates domain self-assessment confidence into issues.
    """

    def qa_node(state: AgentSystemState) -> dict[str, Any]:
        task_id = state.get("task_id", "?")
        logger.info("QA agent starting (critique) | agent=%s task=%s", agent_id, task_id)

        try:
            domain_out = state["domain_outputs"].get(agent_id)
            if not domain_out:
                logger.warning(
                    "QA agent found no domain output | agent=%s task=%s",
                    agent_id, task_id,
                )
                report = QAReport(
                    agent_id=agent_id,
                    verdict="fail",
                    severity="critical",
                    confidence=1.0,
                    issues=[
                        QAIssue(
                            severity="critical",
                            category="completeness",
                            description=f"No output found from domain agent {agent_id}",
                        )
                    ],
                    summary="Missing domain output",
                )
                return {
                    "qa_reports": [report],
                    "audit_log": [
                        {
                            "event": "qa_agent_completed",
                            "agent_id": agent_id,
                            "verdict": "fail",
                            "issue_count": 1,
                            "note": "missing_domain_output",
                        }
                    ],
                }

            if model is not None:
                report = _llm_qa_call(model, agent_id, domain_out, state["original_goal"])
                # Still apply threshold gates on top of LLM critique
                report = _apply_confidence_thresholds(report, domain_out, thresholds)
            else:
                report = _mock_qa_call(
                    agent_id, domain_out, state.get("revision_count", 0), thresholds
                )

            logger.info(
                "QA agent completed (critique) | agent=%s verdict=%s issues=%d",
                agent_id, report.verdict, len(report.issues),
            )

            return {
                "qa_reports": [report],
                "audit_log": [
                    {
                        "event": "qa_agent_completed",
                        "agent_id": agent_id,
                        "verdict": report.verdict,
                        "issue_count": len(report.issues),
                        "mode": "critique",
                    }
                ],
            }
        except Exception as e:
            logger.exception(
                "QA agent failed | agent=%s task=%s error=%s",
                agent_id, task_id, e,
            )
            return {
                "audit_log": [
                    {
                        "event": "qa_agent_error",
                        "agent_id": agent_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                ],
                "error": f"qa_agent:{agent_id}:{e}",
            }

    qa_node.__name__ = f"qa_{agent_id}"
    return qa_node


def _mock_qa_call(
    agent_id: str,
    domain_out: DomainOutput,
    revision: int,
    thresholds: "ConfidenceThresholds | None" = None,
) -> QAReport:
    """Deterministic critique for demo mode, driven by confidence thresholds."""
    from .thresholds import DEFAULT_THRESHOLDS

    th = thresholds or DEFAULT_THRESHOLDS
    issues: list[QAIssue] = []

    # --- Threshold-driven self-assessment gates ---
    conf = domain_out.confidence
    justification = domain_out.confidence_justification or ""

    if conf < th.domain_critical:
        issues.append(
            QAIssue(
                severity="critical",
                category="completeness",
                description=(
                    f"Domain self-assessment confidence {conf:.2f} is below critical "
                    f"threshold {th.domain_critical:.2f}"
                ),
                evidence=justification or None,
                suggested_fix="Re-run with stronger evidence or escalate; output is not reliable",
            )
        )
    elif conf < th.domain_low:
        issues.append(
            QAIssue(
                severity="major",
                category="completeness",
                description=(
                    f"Domain self-assessment confidence {conf:.2f} is below low "
                    f"threshold {th.domain_low:.2f}"
                ),
                evidence=justification or None,
                suggested_fix="Gather additional evidence or refine with targeted feedback",
            )
        )

    # Residual risks declared by the agent itself → minor issues
    for risk in domain_out.metadata.get("residual_risks") or []:
        if risk and conf < th.domain_approve_min:
            issues.append(
                QAIssue(
                    severity="minor",
                    category="completeness",
                    description=f"Residual risk noted by agent: {risk}",
                    suggested_fix="Address or explicitly accept this residual risk",
                )
            )

    # First-cycle simulated consistency issue so the refine loop is exercised
    if revision == 0 and agent_id == "analysis":
        issues.append(
            QAIssue(
                severity="major",
                category="consistency",
                description="Potential contradiction with other domain agents on key claims",
                evidence="Cross-reference required",
                suggested_fix="Align claims with primary sources and other agents",
            )
        )

    # After a successful refine, drop non-critical threshold issues if confidence recovered
    if revision > 0 and domain_out.metadata.get("mode") == "refine":
        if conf >= th.domain_low:
            issues = [
                i for i in issues
                if i.severity == "critical" or i.category == "consistency"
            ]
            # Consistency issues from prior cycle are considered addressed on refine
            issues = [i for i in issues if i.category != "consistency"]

    if issues:
        verdict = "fail" if any(i.severity == "critical" for i in issues) else "conditional"
        severity = max(
            (i.severity for i in issues),
            key=lambda s: {"critical": 3, "major": 2, "minor": 1}[s],
        )
    else:
        verdict = "pass"
        severity = "none"

    return QAReport(
        agent_id=agent_id,
        verdict=verdict,
        severity=severity,
        confidence=0.88,
        issues=issues,
        summary=f"QA critique for {agent_id}: {verdict} ({len(issues)} issues)",
    )


def _llm_qa_call(model, agent_id: str, domain_out: DomainOutput, goal: str) -> QAReport:
    """Real LLM critique with structured QAReport output."""
    system = """You are a rigorous QA critic in a multi-agent system.
Your only job is to find weaknesses, gaps, inconsistencies, and risks in the specialist's output.
Be adversarial but fair. Prefer specific, actionable issues over vague comments.
Return a structured QAReport."""

    human = (
        f"Original goal:\n{goal}\n\n"
        f"Specialist agent: {agent_id}\n"
        f"Self-assessed confidence: {domain_out.confidence}\n"
        f"Justification: {domain_out.confidence_justification or '(none)'}\n\n"
        f"Output to critique:\n{domain_out.content}"
    )

    structured = model.with_structured_output(QAReport)
    report: QAReport = structured.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": human},
    ])
    if report.agent_id != agent_id:
        report = report.model_copy(update={"agent_id": agent_id})
    return report


def _apply_confidence_thresholds(
    report: QAReport,
    domain_out: DomainOutput,
    thresholds: "ConfidenceThresholds | None" = None,
) -> QAReport:
    """
    Overlay deterministic confidence gates on top of an LLM (or other) QA report.
    Ensures low self-assessment cannot silently pass.
    """
    from .thresholds import DEFAULT_THRESHOLDS

    th = thresholds or DEFAULT_THRESHOLDS
    issues = list(report.issues)
    conf = domain_out.confidence
    justification = domain_out.confidence_justification or ""

    def _already_has(severity: str, category: str) -> bool:
        return any(i.severity == severity and i.category == category for i in issues)

    if conf < th.domain_critical and not _already_has("critical", "completeness"):
        issues.append(
            QAIssue(
                severity="critical",
                category="completeness",
                description=(
                    f"Domain self-assessment confidence {conf:.2f} is below critical "
                    f"threshold {th.domain_critical:.2f}"
                ),
                evidence=justification or None,
                suggested_fix="Re-run with stronger evidence or escalate",
            )
        )
    elif conf < th.domain_low and not _already_has("major", "completeness"):
        issues.append(
            QAIssue(
                severity="major",
                category="completeness",
                description=(
                    f"Domain self-assessment confidence {conf:.2f} is below low "
                    f"threshold {th.domain_low:.2f}"
                ),
                evidence=justification or None,
                suggested_fix="Gather additional evidence or refine",
            )
        )

    if not issues:
        return report

    verdict = "fail" if any(i.severity == "critical" for i in issues) else (
        "pass" if report.verdict == "pass" and not issues else "conditional"
    )
    if any(i.severity in ("critical", "major") for i in issues):
        verdict = "fail" if any(i.severity == "critical" for i in issues) else "conditional"

    severity = "none"
    if issues:
        severity = max(
            (i.severity for i in issues),
            key=lambda s: {"critical": 3, "major": 2, "minor": 1, "none": 0}.get(s, 0),
        )

    return report.model_copy(
        update={
            "issues": issues,
            "verdict": verdict,
            "severity": severity,
            "summary": f"QA critique for {report.agent_id}: {verdict} ({len(issues)} issues)",
        }
    )


# ---------------------------------------------------------------------------
# Selective Router — LLM + keyword fallback
# ---------------------------------------------------------------------------

from typing import Literal


class RouterDecision(BaseModel):
    """
    Structured output from the LLM router.
    Validated tightly so the graph never receives garbage agent ids.
    """
    selected_agents: list[str] = Field(
        min_length=1,
        description="List of agent_ids that should run for this goal. Must be a non-empty subset of the available agents.",
    )
    rationale: str = Field(
        min_length=5,
        description="Brief explanation of why these agents were chosen.",
    )
    confidence: float = Field(
        ge=0.0, le=1.0, default=0.8,
        description="Router confidence in the selection.",
    )

    @field_validator("selected_agents")
    @classmethod
    def agents_must_be_non_empty_strings(cls, v: list[str]) -> list[str]:
        cleaned = []
        for item in v:
            if not isinstance(item, str):
                continue
            s = item.strip()
            if s and s not in cleaned:
                cleaned.append(s)
        if not cleaned:
            raise ValueError("selected_agents must contain at least one non-blank agent_id")
        return cleaned

    @field_validator("rationale")
    @classmethod
    def rationale_not_blank(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 5:
            raise ValueError("rationale must be at least 5 characters")
        return v

    def validated_against(self, available: list[str]) -> "RouterDecision":
        """
        Post-parse validation: keep only agents that exist in the catalog.
        Raises ValueError if the intersection is empty.
        """
        valid = [a for a in self.selected_agents if a in available]
        if not valid:
            raise ValueError(
                f"Router selected no valid agents. "
                f"Got {self.selected_agents}, available={available}"
            )
        # Return a new instance with the cleaned list
        return self.model_copy(update={"selected_agents": valid})


# Keyword fallback (used when no LLM is provided or the call fails)
_AGENT_KEYWORDS: dict[str, list[str]] = {
    "research": [
        "research", "market", "competitor", "competitive", "landscape",
        "sources", "data", "survey", "coding assistant", "copilot", "cursor",
    ],
    "analysis": [
        "analysis", "strategic", "compare", "evaluate", "trade-off",
        "insight", "barrier", "adoption", "enterprise",
    ],
    "synthesis": [
        "synthesis", "recommend", "executive", "summary", "conclusion",
        "action", "plan", "produce",
    ],
    # Amigo 21 website page specialists (sheer-test pipeline)
    "home": ["home", "hero", "landing", "homepage", "website", "amigo"],
    "courses": ["course", "courses", "track", "tracks", "curriculum", "lesson"],
    "about": ["about", "mission", "school", "bilingual", "company"],
    "pricing": ["pricing", "price", "tuition", "cost", "plan", "plans"],
    "contact": ["contact", "inquiry", "form", "email", "phone", "message"],
    "faq": ["faq", "question", "questions", "help", "support"],
}


def _keyword_select(
    goal: str,
    available: list[str],
    required_actions: list[dict] | None = None,
) -> list[str]:
    """Deterministic keyword-based selection (fallback)."""
    if required_actions:
        targets = []
        for action in required_actions:
            t = action.get("target")
            if t and t in available and t not in targets:
                targets.append(t)
        if targets:
            return targets

    goal_lower = goal.lower()
    selected = []
    for agent_id in available:
        keywords = _AGENT_KEYWORDS.get(agent_id, [agent_id])
        if any(kw in goal_lower for kw in keywords):
            selected.append(agent_id)

    if not selected:
        selected = list(available)
    return selected


def _llm_select(
    goal: str,
    available: list[str],
    agent_catalog: list[tuple[str, str]],
    router_model,
    required_actions: list[dict] | None = None,
) -> tuple[list[str], str, float]:
    """
    LLM-powered selective router using structured output.

    Returns (selected_agents, rationale, confidence).
    Falls back to keyword selection on any error.
    """
    # Priority 1: targeted revision from QA-Orchestrator
    if required_actions:
        targets = []
        for action in required_actions:
            t = action.get("target")
            if t and t in available and t not in targets:
                targets.append(t)
        if targets:
            return targets, "Revision targets from QA-Orchestrator", 1.0

    catalog_text = "\n".join(
        f"- {aid}: {specialty}" for aid, specialty in agent_catalog if aid in available
    )

    system = """You are a precise routing agent for a multi-agent system.
Given a user goal and a catalog of specialist agents, select ONLY the agents that are necessary to accomplish the goal.
Do not select agents that are irrelevant. Prefer the smallest sufficient set.
Return a structured decision."""

    human = f"""Goal:
{goal}

Available specialist agents:
{catalog_text}

Select the agent_ids that should run."""

    try:
        logger.info("LLM router invoking | available=%s", available)
        structured = router_model.with_structured_output(RouterDecision)
        raw_decision: RouterDecision = structured.invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": human},
        ])

        # Strict post-validation against the live catalog
        decision = raw_decision.validated_against(available)

        logger.info(
            "LLM router succeeded | selected=%s confidence=%.2f",
            decision.selected_agents, decision.confidence,
        )
        return decision.selected_agents, decision.rationale, decision.confidence

    except (ValidationError, ValueError) as e:
        # Structured output was malformed or selected unknown agents
        logger.warning(
            "LLM router validation failed | error=%s | falling back to keywords",
            e,
        )
        selected = _keyword_select(goal, available, required_actions)
        return (
            selected,
            f"LLM router validation failed ({type(e).__name__}: {e}); used keyword fallback",
            0.4,
        )
    except Exception as e:
        # Network / provider / unexpected errors
        logger.exception(
            "LLM router failed | error=%s | falling back to keywords", e
        )
        selected = _keyword_select(goal, available, required_actions)
        return (
            selected,
            f"LLM router failed ({type(e).__name__}); used keyword fallback",
            0.5,
        )


def _select_agents(
    goal: str,
    available: list[str],
    agent_catalog: list[tuple[str, str]],
    router_model=None,
    required_actions: list[dict] | None = None,
) -> tuple[list[str], str, str]:
    """
    Unified entry point.

    Returns (selected_agents, rationale, routing_method).
    """
    if router_model is not None:
        selected, rationale, _ = _llm_select(
            goal, available, agent_catalog, router_model, required_actions
        )
        return selected, rationale, "llm"

    selected = _keyword_select(goal, available, required_actions)
    return selected, "keyword matching", "keyword"


# ---------------------------------------------------------------------------
# Main Orchestrator factory (selective router)
# ---------------------------------------------------------------------------

def make_main_orchestrator(
    available_agent_ids: list[str],
    agent_catalog: list[tuple[str, str]] | None = None,
    router_model=None,
):
    """
    Factory that closes over the full catalog of domain agent ids
    and an optional LLM router model.

    Args:
        available_agent_ids: list of agent_id strings the graph knows about
        agent_catalog: list of (agent_id, specialty) for richer LLM prompts
        router_model: any LangChain chat model that supports .with_structured_output
                      If None, falls back to keyword routing (no API key required)
    """
    catalog = agent_catalog or [(aid, aid) for aid in available_agent_ids]

    def main_orchestrator(state: AgentSystemState) -> dict[str, Any]:
        """
        Top-level planner + selective router.

        - Decides which domain agents are relevant for the current goal / revision.
        - Writes `selected_agents` into state so the fan-out only launches those agents.
        """
        revision = state.get("revision_count", 0)
        decision = state.get("quality_decision")
        goal = state["original_goal"]
        task_id = state.get("task_id", "?")

        logger.info(
            "Orchestrator starting | task=%s revision=%s",
            task_id, revision,
        )

        try:
            required_actions = decision.required_actions if decision else None
            selected, rationale, method = _select_agents(
                goal=goal,
                available=available_agent_ids,
                agent_catalog=catalog,
                router_model=router_model,
                required_actions=required_actions,
            )

            event_name = (
                "orchestrator_planned"
                if (revision == 0 and not decision)
                else "orchestrator_revision"
            )

            logger.info(
                "Orchestrator routing decision | method=%s selected=%s rationale=%s",
                method, selected, rationale[:80] if rationale else "",
            )

            return {
                "current_phase": "domain_execution",
                "selected_agents": selected,
                "audit_log": [
                    {
                        "event": event_name,
                        "goal": goal if event_name == "orchestrator_planned" else None,
                        "revision": revision,
                        "selected_agents": selected,
                        "available_agents": available_agent_ids,
                        "routing": method,
                        "rationale": rationale,
                        "actions": required_actions or [],
                    }
                ],
            }
        except Exception as e:
            logger.exception(
                "Orchestrator failed | task=%s error=%s", task_id, e
            )
            # Safe fallback: run everyone
            return {
                "current_phase": "domain_execution",
                "selected_agents": list(available_agent_ids),
                "audit_log": [
                    {
                        "event": "orchestrator_error",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "fallback_selected": list(available_agent_ids),
                    }
                ],
                "error": f"orchestrator:{e}",
            }

    main_orchestrator.__name__ = "orchestrator"
    return main_orchestrator


# ---------------------------------------------------------------------------
# QA-Orchestrator (system-level quality controller)
# ---------------------------------------------------------------------------

def make_qa_orchestrator(thresholds=None):
    """Factory so the QA-Orchestrator can close over confidence thresholds."""
    from .thresholds import DEFAULT_THRESHOLDS

    th = thresholds or DEFAULT_THRESHOLDS

    def qa_orchestrator(state: AgentSystemState) -> dict[str, Any]:
        """
        Aggregates all QA reports, applies confidence thresholds,
        detects cross-agent conflicts, and issues a final disposition.
        """
        task_id = state.get("task_id", "?")
        logger.info("QA-Orchestrator starting | task=%s", task_id)

        try:
            all_reports = state.get("qa_reports", [])
            n_domain = len(state.get("domain_outputs", {})) or 1
            reports = all_reports[-n_domain:] if all_reports else []

            revision = state.get("revision_count", 0)
            max_rev = state.get("max_revisions", 3)
            domain_outputs = state.get("domain_outputs", {})

            critical_issues = []
            major_issues = []
            conflicts = []

            for r in reports:
                for issue in r.issues:
                    if issue.severity == "critical":
                        critical_issues.append(f"{r.agent_id}: {issue.description}")
                    elif issue.severity == "major":
                        major_issues.append(f"{r.agent_id}: {issue.description}")
                    if issue.category == "consistency":
                        conflicts.append(f"{r.agent_id}: {issue.description}")

            # --- Confidence threshold gates ---
            low_domain = []
            for aid, dout in domain_outputs.items():
                if dout.confidence < th.domain_approve_min:
                    low_domain.append(f"{aid}:{dout.confidence:.2f}")

            avg_qa_conf = (
                sum(r.confidence for r in reports) / len(reports) if reports else 0.0
            )
            qa_conf_ok = avg_qa_conf >= th.qa_approve_min

            # Decision policy (thresholds participate)
            if critical_issues:
                disposition = "reject" if revision >= max_rev else "escalate"
                score = 0.2
                rationale = "Critical issues detected: " + "; ".join(critical_issues[:3])
                actions = []
            elif conflicts or major_issues:
                if revision >= max_rev:
                    disposition = "escalate"
                    score = 0.45
                    rationale = (
                        f"Max revisions ({max_rev}) reached with unresolved "
                        "major/consistency issues."
                    )
                    actions = []
                else:
                    disposition = "revise"
                    score = 0.55
                    rationale = (
                        "Major issues or cross-agent conflicts require targeted revision."
                    )
                    actions = [
                        {
                            "target": r.agent_id,
                            "instruction": "Address consistency and major findings from QA.",
                        }
                        for r in reports
                        if r.verdict != "pass"
                    ]
            elif low_domain:
                # No hard issues, but domain self-assessment below approve floor
                if revision >= max_rev:
                    disposition = "escalate"
                    score = 0.50
                    rationale = (
                        f"Domain confidence below approve threshold "
                        f"({th.domain_approve_min:.2f}): {', '.join(low_domain)}. "
                        f"Max revisions reached."
                    )
                    actions = []
                else:
                    disposition = "revise"
                    score = 0.60
                    rationale = (
                        f"Domain confidence below approve threshold "
                        f"({th.domain_approve_min:.2f}): {', '.join(low_domain)}."
                    )
                    actions = [
                        {
                            "target": aid.split(":")[0],
                            "instruction": (
                                f"Raise self-assessed confidence above "
                                f"{th.domain_approve_min:.2f} by strengthening evidence."
                            ),
                        }
                        for aid in low_domain
                    ]
            elif not qa_conf_ok:
                disposition = "revise" if revision < max_rev else "escalate"
                score = 0.65
                rationale = (
                    f"Average QA critic confidence {avg_qa_conf:.2f} is below "
                    f"threshold {th.qa_approve_min:.2f}."
                )
                actions = [
                    {
                        "target": r.agent_id,
                        "instruction": "Re-examine output; critic confidence was insufficient.",
                    }
                    for r in reports
                ] if revision < max_rev else []
            else:
                disposition = "approve"
                score = max(th.quality_approve_min, 0.92)
                rationale = (
                    "All domain agents passed QA; confidence thresholds satisfied "
                    f"(domain >= {th.domain_approve_min:.2f}, "
                    f"qa_avg >= {th.qa_approve_min:.2f})."
                )
                actions = []

            decision = QualityDecision(
                disposition=disposition,
                rationale=rationale,
                quality_score=score,
                cross_agent_conflicts=conflicts,
                required_actions=actions,
                revision_count_at_decision=revision,
            )

            new_revision = revision + 1 if disposition == "revise" else revision

            phase_map = {
                "approve": "finalize",
                "revise": "revision",
                "escalate": "escalated",
                "reject": "rejected",
            }

            logger.info(
                "QA-Orchestrator decision | disposition=%s score=%.2f revision=%s "
                "conflicts=%d low_domain=%s qa_avg=%.2f",
                disposition, score, new_revision, len(conflicts),
                low_domain, avg_qa_conf,
            )

            return {
                "quality_decision": decision,
                "revision_count": new_revision,
                "current_phase": phase_map[disposition],
                "audit_log": [
                    {
                        "event": "qa_orchestrator_decision",
                        "disposition": disposition,
                        "quality_score": score,
                        "revision": new_revision,
                        "conflicts": len(conflicts),
                        "low_domain_confidence": low_domain,
                        "avg_qa_confidence": round(avg_qa_conf, 3),
                        "thresholds": {
                            "domain_approve_min": th.domain_approve_min,
                            "qa_approve_min": th.qa_approve_min,
                        },
                    }
                ],
            }
        except Exception as e:
            logger.exception(
                "QA-Orchestrator failed | task=%s error=%s", task_id, e
            )
            return {
                "current_phase": "rejected",
                "error": f"qa_orchestrator:{e}",
                "audit_log": [
                    {
                        "event": "qa_orchestrator_error",
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                ],
            }

    qa_orchestrator.__name__ = "qa_orchestrator"
    return qa_orchestrator


# Backwards-compatible default instance (uses DEFAULT_THRESHOLDS)
qa_orchestrator = make_qa_orchestrator()


# ---------------------------------------------------------------------------
# Terminal nodes
# ---------------------------------------------------------------------------

def finalize_node(state: AgentSystemState) -> dict[str, Any]:
    decision = state.get("quality_decision")
    logger.info(
        "Finalized | disposition=%s score=%s task=%s",
        decision.disposition if decision else None,
        decision.quality_score if decision else None,
        state.get("task_id"),
    )
    return {
        "current_phase": "finalize",
        "audit_log": [
            {
                "event": "finalized",
                "quality_score": decision.quality_score if decision else None,
                "disposition": decision.disposition if decision else None,
            }
        ],
    }


def escalate_node(state: AgentSystemState) -> dict[str, Any]:
    logger.warning(
        "Escalated to human | task=%s", state.get("task_id")
    )
    return {
        "current_phase": "escalated",
        "audit_log": [{"event": "escalated_to_human", "task_id": state["task_id"]}],
    }


def reject_node(state: AgentSystemState) -> dict[str, Any]:
    decision = state.get("quality_decision")
    logger.error(
        "Rejected | reason=%s task=%s",
        decision.rationale if decision else "unknown",
        state.get("task_id"),
    )
    return {
        "current_phase": "rejected",
        "error": decision.rationale if decision else "Rejected by QA-Orchestrator",
        "audit_log": [{"event": "rejected", "reason": decision.rationale if decision else "unknown"}],
    }
