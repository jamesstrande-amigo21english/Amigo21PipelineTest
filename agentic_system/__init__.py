"""
Full-scale multi-agent system with:
  - Configurable N domain agents
  - 1 Main Orchestrator
  - N dedicated QA agents (1:1)
  - 1 QA-Orchestrator

Built on LangGraph with proper state schema, parallel fan-out,
revision loops, and checkpointer support.
"""

from .graph import build_agent_system, create_initial_state
from .state import (
    AgentSystemState,
    DomainOutput,
    QAReport,
    QualityDecision,
    QAIssue,
)
from .agents import RouterDecision, DomainAgentResponse
from .thresholds import ConfidenceThresholds, DEFAULT_THRESHOLDS

__all__ = [
    "build_agent_system",
    "create_initial_state",
    "AgentSystemState",
    "DomainOutput",
    "QAReport",
    "QualityDecision",
    "QAIssue",
    "RouterDecision",
    "DomainAgentResponse",
    "ConfidenceThresholds",
    "DEFAULT_THRESHOLDS",
]
