"""Smoke test: every behavior imports + registers without errors.

Does not call any external services.
"""
from __future__ import annotations

import os


def test_imports_and_registers():
    # Set fake env so config.py doesn't fail at import.
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
    os.environ.setdefault("MONID_API_KEY", "test-key")

    from research_agent.behaviors import (
        ALL_BEHAVIORS,
        behavior_count,
        register_all,
    )

    # ALL_BEHAVIORS is module-level only (no factory instances).
    # behavior_count() includes factory-built decomposer/planner/synthesizer.
    assert behavior_count() == len(ALL_BEHAVIORS) + 3

    # Default-cap registration.
    registry = register_all()
    names = [b.name for b in registry]
    assert len(names) == behavior_count()
    assert len(names) == len(set(names)), f"duplicate behavior names: {names}"

    # Re-register: idempotent.
    registry2 = register_all()
    assert len(registry2) == behavior_count()

    # Custom-cap registration works.
    registry3 = register_all(
        max_strategies=2,
        max_tasks_per_strategy=1,
        memo_mode="short",
    )
    assert len(registry3) == behavior_count()


def test_memo_modes_accepted():
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
    os.environ.setdefault("MONID_API_KEY", "test-key")
    from research_agent.behaviors import register_all

    for mode in ("short", "auto", "long", "detailed"):
        registry = register_all(memo_mode=mode)
        names = [b.name for b in registry]
        assert "synthesizer" in names


def test_memo_mode_invalid():
    import pytest
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
    os.environ.setdefault("MONID_API_KEY", "test-key")
    from research_agent.behaviors.synthesizer import make_synthesizer

    with pytest.raises(ValueError):
        make_synthesizer(memo_mode="medium")


def test_behavior_subscriptions():
    """Every event type emitted should have a subscriber somewhere."""
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
    os.environ.setdefault("MONID_API_KEY", "test-key")
    from research_agent.behaviors import register_all
    from activegraph import get_registry

    register_all()
    subs: set[str] = set()
    for b in get_registry():
        for t in b.on or []:
            subs.add(t)
    expected_subs = {
        "goal.created",
        "strategy.proposed",
        "strategy.needs_more_tasks",
        "strategy.complete",
        "strategy.capped",
        "strategy.abandoned",
        "task.proposed",
        "task.candidates.ready",
        "endpoint.selected",
        "endpoint.inspected",
        "endpoint.input_ready",
        "task.results.ready",
        "research.run.completed",
    }
    missing = expected_subs - subs
    assert not missing, f"no behavior subscribes to: {missing}"


def test_types_schemas():
    from research_agent.types import (
        ExtractedClaims,
        InputSpec,
        MemoOutput,
        RouteChoice,
        StrategyOutcome,
        StrategyVerdict,
        make_decomposition_schema,
        make_task_plan_schema,
    )

    Decomposition3 = make_decomposition_schema(3)
    Decomposition3.model_validate(
        {"strategies": [{"name": "n", "rationale": "r"}]}
    )

    TaskPlan2 = make_task_plan_schema(2)
    TaskPlan2.model_validate(
        {"tasks": [{"description": "d", "discover_queries": ["q"]}]}
    )

    RouteChoice.model_validate({"provider": "apify", "endpoint": "/x"})
    InputSpec.model_validate({"input": {"q": "x"}})
    StrategyVerdict.model_validate({"verdict": "complete"})
    MemoOutput.model_validate({"summary": "s"})
    ExtractedClaims.model_validate({"claims": []})

    # `capped` is now a legal outcome status.
    StrategyOutcome.model_validate(
        {"strategy_id": "x", "status": "capped", "one_line_summary": "y"}
    )


def test_factory_caps_enforced():
    """Pydantic should reject schemas exceeding the configured cap."""
    import pytest
    from research_agent.types import (
        make_decomposition_schema,
        make_task_plan_schema,
    )

    Decomposition2 = make_decomposition_schema(2)
    with pytest.raises(Exception):
        Decomposition2.model_validate(
            {
                "strategies": [
                    {"name": "a", "rationale": "r"},
                    {"name": "b", "rationale": "r"},
                    {"name": "c", "rationale": "r"},
                ]
            }
        )

    TaskPlan1 = make_task_plan_schema(1)
    with pytest.raises(Exception):
        TaskPlan1.model_validate(
            {
                "tasks": [
                    {"description": "d1", "discover_queries": ["q"]},
                    {"description": "d2", "discover_queries": ["q"]},
                ]
            }
        )
