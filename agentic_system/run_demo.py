"""
MOCK-mode end-to-end demo for the Amigo 21 website-build pipeline.

No LLM API keys required. Domain agents emit deterministic page drafts;
QA + QA-Orchestrator use DEFAULT_THRESHOLDS unchanged.

Sheer-test target: Amigo21PipelineTest (local scratch under
/workspace/amigo21-pipeline/). Does not touch Webflow, Cloud SQL,
payments, or live contact POST.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from . import (
    ConfidenceThresholds,
    DEFAULT_THRESHOLDS,
    build_agent_system,
    create_initial_state,
)
from .logging_config import setup_logging, get_logger

# Package root = /workspace/amigo21-pipeline when run as module
ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
LOG_PATH = RUNS / "mock-run-1.log"

# Six marketing pages, each with a dedicated 1:1 QA agent via graph wiring.
DOMAIN_AGENTS: list[tuple[str, str]] = [
    ("home", "Home page — hero, value prop, CTAs for Amigo 21 English"),
    ("courses", "Courses/Tracks page — 12 courses across 3 tracks"),
    ("about", "About page — school mission and bilingual offering"),
    (
        "pricing",
        "Pricing page — informational tuition matrix "
        "(NONPRODUCTION only; not a live checkout)",
    ),
    (
        "contact",
        "Contact page — inquiry form documenting existing POST /api/contact "
        "(no real call in mock)",
    ),
    ("faq", "FAQ page — common learner questions"),
]

GOAL = (
    "Build Amigo 21 English marketing website page drafts for home, courses/tracks, "
    "about, pricing (informational/nonproduction only), contact (document existing "
    "POST /api/contact — do not call it), and FAQ. Emit structured page content for "
    "the sheer-test pipeline repo; do not deploy, push, or touch Webflow/Cloud SQL/"
    "payments/live services."
)


def _configure_file_logging(log_path: Path) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    setup_logging(level=logging.INFO)
    root = logging.getLogger("agentic_system")
    # Avoid duplicate file handlers on re-entry
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(
            log_path.resolve()
        ):
            return
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(fh)


def _serialize_result(result: dict) -> dict:
    out: dict = {
        "task_id": result.get("task_id"),
        "current_phase": result.get("current_phase"),
        "revision_count": result.get("revision_count"),
        "selected_agents": result.get("selected_agents"),
        "error": result.get("error"),
    }
    qd = result.get("quality_decision")
    if qd is not None:
        out["quality_decision"] = qd.model_dump() if hasattr(qd, "model_dump") else qd
    domain = {}
    for aid, dout in (result.get("domain_outputs") or {}).items():
        domain[aid] = dout.model_dump() if hasattr(dout, "model_dump") else dout
    out["domain_outputs"] = domain
    reports = []
    for r in result.get("qa_reports") or []:
        reports.append(r.model_dump() if hasattr(r, "model_dump") else r)
    out["qa_reports"] = reports
    out["audit_log"] = result.get("audit_log") or []
    return out


def _write_page_artifacts(result: dict, run_dir: Path) -> None:
    """Emit per-page Markdown + a summary JSON (sheer-test artifact format)."""
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    domain = result.get("domain_outputs") or {}
    for aid, dout in domain.items():
        content = dout.content if hasattr(dout, "content") else dout.get("content", "")
        (pages_dir / f"{aid}.md").write_text(str(content), encoding="utf-8")
    summary = _serialize_result(result)
    (run_dir / "result.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> int:
    _configure_file_logging(LOG_PATH)
    logger = get_logger("run_demo")
    logger.info("MOCK E2E starting | log=%s", LOG_PATH)
    logger.info("Thresholds (unchanged): %s", DEFAULT_THRESHOLDS)

    thresholds = ConfidenceThresholds()  # same as DEFAULT; do not weaken
    checkpointer = InMemorySaver()
    graph = build_agent_system(
        domain_agents=DOMAIN_AGENTS,
        max_revisions=2,
        checkpointer=checkpointer,
        router_model=None,   # keyword / mock — no API key
        domain_model=None,   # mock draft/refine
        qa_model=None,       # mock critique
        thresholds=thresholds,
    )

    state = create_initial_state(goal=GOAL, max_revisions=2)
    config = {"configurable": {"thread_id": state["task_id"]}}
    logger.info("Invoke graph | task_id=%s agents=%s", state["task_id"], [a for a, _ in DOMAIN_AGENTS])

    result = graph.invoke(state, config)

    run_dir = RUNS / f"mock-run-1-{state['task_id']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_page_artifacts(result, run_dir)
    # Stable alias for the latest mock run outputs
    latest = RUNS / "mock-run-1"
    if latest.exists() or latest.is_symlink():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            # replace directory alias marker
            import shutil
            shutil.rmtree(latest)
    # Copy tree into stable path
    import shutil
    shutil.copytree(run_dir, latest)

    phase = result.get("current_phase")
    qd = result.get("quality_decision")
    disp = qd.disposition if qd else None
    score = qd.quality_score if qd else None

    logger.info(
        "MOCK E2E finished | phase=%s disposition=%s score=%s revisions=%s",
        phase, disp, score, result.get("revision_count"),
    )

    # Per-page QA outcomes (latest report per agent)
    latest_by_agent: dict[str, object] = {}
    for r in result.get("qa_reports") or []:
        latest_by_agent[r.agent_id] = r
    for aid, _ in DOMAIN_AGENTS:
        r = latest_by_agent.get(aid)
        if r is None:
            logger.warning("QA missing | page=%s", aid)
        else:
            logger.info(
                "QA outcome | page=%s verdict=%s severity=%s issues=%d conf=%.2f",
                aid, r.verdict, r.severity, len(r.issues), r.confidence,
            )

    print("=" * 60)
    print("MOCK E2E RESULT")
    print(f"  phase:        {phase}")
    print(f"  disposition:  {disp}")
    print(f"  quality:      {score}")
    print(f"  revisions:    {result.get('revision_count')}")
    print(f"  log:          {LOG_PATH}")
    print(f"  outputs:      {latest}")
    print("=" * 60)
    for aid, _ in DOMAIN_AGENTS:
        r = latest_by_agent.get(aid)
        dout = (result.get("domain_outputs") or {}).get(aid)
        dconf = dout.confidence if dout else None
        if r is None:
            print(f"  {aid:10s}  QA=MISSING")
        else:
            print(
                f"  {aid:10s}  QA={r.verdict:12s} sev={r.severity:8s} "
                f"issues={len(r.issues)} domain_conf={dconf}"
            )

    ok = phase == "finalize" and disp == "approve"
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
