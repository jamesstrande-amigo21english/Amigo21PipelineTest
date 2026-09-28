"""
LangGraph construction for the full multi-agent system.

Architecture:
  Main Orchestrator
       │
       ├── Domain Agent 1 ──► QA-1
       ├── Domain Agent 2 ──► QA-2
       └── Domain Agent N ──► QA-N
                │
                ▼
         QA-Orchestrator
                │
       ┌────────┼────────┐
       ▼        ▼        ▼
   finalize  revise  escalate/reject
"""

from __future__ import annotations

import uuid
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Send

from .state import AgentSystemState
from .agents import (
    make_domain_agent,
    make_qa_agent,
    make_main_orchestrator,
    make_qa_orchestrator,
    finalize_node,
    escalate_node,
    reject_node,
)
from .thresholds import ConfidenceThresholds, DEFAULT_THRESHOLDS


def build_agent_system(
    domain_agents: list[tuple[str, str]],
    max_revisions: int = 3,
    checkpointer=None,
    router_model=None,
    domain_model=None,
    qa_model=None,
    thresholds: ConfidenceThresholds | None = None,
):
    """
    Build the full multi-agent graph with selective routing,
    draft → critique → refine loop, and confidence thresholds.

    Args:
        domain_agents: list of (agent_id, specialty) tuples.
        max_revisions: circuit-breaker for revision loops
        checkpointer: optional LangGraph checkpointer
        router_model: optional chat model for selective routing
        domain_model: optional chat model for DRAFT + REFINE (e.g. Claude Opus 5)
        qa_model: optional chat model for CRITIQUE (e.g. GPT-5.6)
        thresholds: confidence gates (defaults to DEFAULT_THRESHOLDS)

    Flow per agent:
        domain (draft) → qa (critique) → qa_orchestrator
            ↓ revise
        domain (refine using critique) → qa (re-critique) → ...

    Returns:
        Compiled LangGraph application
    """
    if not domain_agents:
        raise ValueError("At least one domain agent is required")

    th = thresholds or DEFAULT_THRESHOLDS
    available_ids = [aid for aid, _ in domain_agents]

    builder = StateGraph(AgentSystemState)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    orchestrator_fn = make_main_orchestrator(
        available_agent_ids=available_ids,
        agent_catalog=domain_agents,
        router_model=router_model,
    )
    builder.add_node("orchestrator", orchestrator_fn)

    # Domain agents (draft on first pass, refine on revision)
    domain_node_names = []
    for agent_id, specialty in domain_agents:
        node_fn = make_domain_agent(agent_id, specialty, model=domain_model)
        node_name = f"domain_{agent_id}"
        builder.add_node(node_name, node_fn)
        domain_node_names.append(node_name)

    # Dedicated QA agents — the CRITIQUE step (threshold-aware)
    qa_node_names = []
    for agent_id, _ in domain_agents:
        node_fn = make_qa_agent(agent_id, model=qa_model, thresholds=th)
        node_name = f"qa_{agent_id}"
        builder.add_node(node_name, node_fn)
        qa_node_names.append(node_name)

    builder.add_node("qa_orchestrator", make_qa_orchestrator(th))
    builder.add_node("finalize", finalize_node)
    builder.add_node("escalate", escalate_node)
    builder.add_node("reject", reject_node)

    # ------------------------------------------------------------------
    # Edges — selective fan-out
    # ------------------------------------------------------------------
    builder.add_edge(START, "orchestrator")

    def fanout_domains(state: AgentSystemState):
        """
        Selective router fan-out.
        Only launches the domain agents listed in state["selected_agents"].
        Falls back to the full catalog if the orchestrator forgot to set it.
        """
        selected = state.get("selected_agents") or available_ids
        # Guard against unknown ids
        selected = [aid for aid in selected if aid in available_ids]
        if not selected:
            selected = available_ids  # never run zero agents

        return [Send(f"domain_{aid}", state) for aid in selected]

    builder.add_conditional_edges("orchestrator", fanout_domains)

    # Each domain agent goes to its own QA agent
    for agent_id, _ in domain_agents:
        builder.add_edge(f"domain_{agent_id}", f"qa_{agent_id}")

    # All QA agents converge to the QA-Orchestrator.
    # LangGraph only schedules the QA nodes that were actually reached
    # by a preceding domain node, so unselected agents never produce QA reports.
    for qa_name in qa_node_names:
        builder.add_edge(qa_name, "qa_orchestrator")

    # QA-Orchestrator decides the final routing
    def route_after_qa(state: AgentSystemState) -> Literal["finalize", "orchestrator", "escalate", "reject"]:
        decision = state.get("quality_decision")
        if decision is None:
            return "reject"

        disp = decision.disposition
        if disp == "approve":
            return "finalize"
        if disp == "revise":
            return "orchestrator"
        if disp == "escalate":
            return "escalate"
        return "reject"

    builder.add_conditional_edges(
        "qa_orchestrator",
        route_after_qa,
        {
            "finalize": "finalize",
            "orchestrator": "orchestrator",
            "escalate": "escalate",
            "reject": "reject",
        },
    )

    builder.add_edge("finalize", END)
    builder.add_edge("escalate", END)
    builder.add_edge("reject", END)

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
    if checkpointer is None:
        checkpointer = InMemorySaver()

    graph = builder.compile(checkpointer=checkpointer)

    # Attach metadata for convenience
    graph._domain_agents = domain_agents  # type: ignore
    graph._available_ids = available_ids  # type: ignore
    graph._max_revisions = max_revisions  # type: ignore

    return graph


def create_initial_state(
    goal: str,
    task_id: str | None = None,
    max_revisions: int = 3,
) -> AgentSystemState:
    """Helper to create a clean starting state."""
    return {
        "messages": [],
        "original_goal": goal,
        "task_id": task_id or str(uuid.uuid4())[:12],
        "domain_outputs": {},
        "qa_reports": [],
        "revision_count": 0,
        "max_revisions": max_revisions,
        "current_phase": "planning",
        "audit_log": [],
    }
