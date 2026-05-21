"""L5 strategy_evaluator: decide if a strategy is done / needs more / abandoned.

Fires on `task.results.ready`. Only evaluates when ALL tasks for the
strategy are terminal (complete or failed); otherwise no-ops and
waits for the next tick.

The LLM sees the strategy's name + rationale, the prior tasks and
their results, and decides:
  - complete: strategy is sufficiently answered.
  - needs_more_tasks: with `feedback` describing what's still missing.
  - abandon: with `reason` why the strategy can't be completed.

Hard caps:
  - round_count >= max_rounds (3) forces complete.
  - budget exhausted forces abandon.
"""
from __future__ import annotations

from activegraph import llm_behavior

from ..types import StrategyVerdict
from ._helpers import budget_exhausted, view_objects


MAX_ROUNDS = 3  # max planning rounds per strategy (initial + follow-ups)


@llm_behavior(
    name="strategy_evaluator",
    on=["task.results.ready"],
    description=(
        "You are the STRATEGY EVALUATOR. A strategy is a research "
        "direction; it has been broken into 1+ tasks and some of them "
        "have now completed.\n\n"
        "Look at the strategy's name and rationale, then the tasks and "
        "the claims those tasks produced (visible in the view block). "
        "Decide:\n"
        "  - complete: the strategy's intent is sufficiently addressed. "
        "    Provide a one-line summary in `feedback`.\n"
        "  - needs_more_tasks: more tasks would meaningfully improve "
        "    coverage. Put the specific GAP in `feedback`; strategy_planner "
        "    will use it to propose follow-up tasks.\n"
        "  - abandon: no reasonable additional tasks would help. Explain "
        "    why in `reason`.\n\n"
        "Default to `complete` unless there is a clear, concrete gap. "
        "Avoid endless follow-up loops."
    ),
    output_schema=StrategyVerdict,
    creates=[],
    deterministic=True,
)
def strategy_evaluator(event, graph, ctx, llm_output: StrategyVerdict):
    sid = event.payload["strategy_id"]
    strat = graph.get_object(sid)
    if strat is None or strat.data.get("status") != "active":
        return  # already terminal

    # Only evaluate when every task for this strategy is terminal.
    tasks = [
        t for t in view_objects(ctx, "task") if t.data.get("strategy_id") == sid
    ]
    if any(t.data.get("status") == "pending" for t in tasks):
        return

    rounds = int(strat.data.get("round_count", 0))

    # Force-terminal overrides on verdict.
    force_complete = rounds >= MAX_ROUNDS
    force_abandon = budget_exhausted(ctx)

    if force_abandon:
        verdict = "abandon"
        feedback = "budget exhausted"
    elif force_complete:
        verdict = "complete"
        feedback = "max rounds reached; completing with available evidence"
    else:
        verdict = llm_output.verdict
        feedback = llm_output.feedback or llm_output.reason

    if verdict == "complete":
        graph.patch_object(sid, {"status": "complete"})
        graph.emit("strategy.complete", {"strategy_id": sid, "summary": feedback})
    elif verdict == "abandon":
        graph.patch_object(
            sid, {"status": "abandoned", "abandon_reason": feedback or "abandoned"}
        )
        graph.emit("strategy.abandoned", {"strategy_id": sid, "reason": feedback})
    else:  # needs_more_tasks
        graph.emit(
            "strategy.needs_more_tasks",
            {
                "strategy_id": sid,
                "feedback": feedback,
                "prior_task_ids": [t.id for t in tasks],
            },
        )
