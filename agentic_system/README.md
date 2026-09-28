# Full-Scale Multi-Agent System (LangGraph)

Production-oriented skeleton implementing the architecture:

```
Main Orchestrator
       │
       ├── Domain Agent 1 ──► QA-1
       ├── Domain Agent 2 ──► QA-2
       └── Domain Agent N ──► QA-N
                │
                ▼
         QA-Orchestrator
                │
     ┌──────────┼──────────┐
     ▼          ▼          ▼
 finalize    revise    escalate / reject
```

## Features

- Configurable number of domain agents (MVP size)
- 1 Main Orchestrator with **selective LLM router**
- 1 dedicated QA agent per domain agent
- 1 QA-Orchestrator with explicit disposition policy
- Parallel fan-out of only the selected domain + QA stages
- Revision loops with circuit breaker (`max_revisions`)
- Full LangGraph state schema with proper reducers
- Checkpointer support (InMemory for demo, Postgres-ready)
- Audit log + quality decision trail
- Time-travel / state history via checkpoints

## Quick Start

```bash
cd /home/workdir/artifacts
python -m agentic_system.run_demo
```

## Selective LLM Router

The Main Orchestrator can route using either:

1. **Keyword fallback** (default, no API key needed)
2. **LLM structured output** (pass any LangChain chat model)

```python
from langchain_openai import ChatOpenAI
# or from langchain_anthropic import ChatAnthropic

router_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

graph = build_agent_system(
    domain_agents=domain_agents,
    router_model=router_model,          # enables LLM routing
    checkpointer=checkpointer,
)
```

The LLM returns a `RouterDecision`:

```python
class RouterDecision(BaseModel):
    selected_agents: list[str]
    rationale: str
    confidence: float
```

On any LLM failure the system automatically falls back to keyword routing.

## Implementation Code Snippet

Full production-style wiring: dual frontier models, confidence thresholds,
selective router, and the draft → critique → refine loop.

```python
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver  # or PostgresSaver

from agentic_system import (
    build_agent_system,
    create_initial_state,
    ConfidenceThresholds,
)

# --- Models (amplification pair) ---
domain_model = ChatAnthropic(
    model="claude-opus-5-20250514",   # draft + refine
    temperature=0.3,
)
qa_model = ChatOpenAI(
    model="gpt-5.6",                 # critique
    temperature=0.4,
)
router_model = ChatOpenAI(
    model="gpt-5.6-mini",            # cheap selective routing
    temperature=0,
)

# --- Confidence gates ---
thresholds = ConfidenceThresholds(
    domain_critical=0.50,      # below → critical QA issue
    domain_low=0.70,           # below → major QA issue
    domain_approve_min=0.75,   # must clear to auto-approve
    qa_approve_min=0.80,       # avg critic confidence floor
    quality_approve_min=0.85,
)

# --- MVP specialist catalog ---
domain_agents = [
    ("research", "Market & Competitive Research"),
    ("analysis", "Strategic Analysis"),
    ("synthesis", "Executive Synthesis & Recommendations"),
]

# --- Build graph ---
checkpointer = InMemorySaver()
graph = build_agent_system(
    domain_agents=domain_agents,
    max_revisions=2,
    checkpointer=checkpointer,
    domain_model=domain_model,   # Claude Opus 5 → draft + refine
    qa_model=qa_model,           # GPT-5.6 → critique
    router_model=router_model,   # selective fan-out
    thresholds=thresholds,
)

# --- Run ---
goal = (
    "Produce a competitive analysis of three leading AI coding assistants "
    "focused on enterprise adoption barriers."
)
state = create_initial_state(goal=goal, max_revisions=2)
config = {"configurable": {"thread_id": state["task_id"]}}

result = graph.invoke(state, config)

print(result["current_phase"])                 # finalize | escalated | rejected
print(result["quality_decision"].disposition)  # approve | revise | escalate | reject
print(result["quality_decision"].quality_score)
for aid, out in result["domain_outputs"].items():
    print(f"{aid}: conf={out.confidence:.2f} | {out.confidence_justification[:80]}")
```

### What this runs

```
Orchestrator (router) → selected domain agents
        ↓
Domain DRAFT (Opus 5) + structured self-assessment
        ↓
QA CRITIQUE (GPT-5.6) + confidence threshold gates
        ↓
QA-Orchestrator (thresholds + policy)
        ↓ revise?
Domain REFINE (Opus 5, with QA feedback) → re-critique → approve
```

## Extending to Real LLMs

Model injection is already supported via `domain_model`, `qa_model`, and
`router_model`. The factories call `.with_structured_output(...)` for
`DomainAgentResponse`, `QAReport`, and `RouterDecision`.

## Recommended Production Checkpointer

```python
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

pool = ConnectionPool(
    conninfo="postgresql://user:pass@host/db",
    max_size=20,
    kwargs={"autocommit": True, "row_factory": dict_row},
)
checkpointer = PostgresSaver(pool)
checkpointer.setup()

graph = build_agent_system(domain_agents=..., checkpointer=checkpointer)
```

## File Layout

```
agentic_system/
├── __init__.py
├── state.py          # TypedDict + Pydantic models
├── agents.py         # All agent node functions
├── graph.py          # Graph construction + routing
├── run_demo.py       # Runnable end-to-end demo
└── README.md
```
