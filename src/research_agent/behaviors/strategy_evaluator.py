"""L5 strategy_evaluator: decide if a strategy is done / needs more / capped / abandoned.

Fires on `task.results.ready`. Only evaluates when ALL tasks for the
strategy are terminal; otherwise no-ops.

Four possible outcomes:
  - complete   : LLM judged the strategy sufficiently answered.
  - needs_more : LLM wants follow-up tasks (one more round).
  - capped     : External cap (budget or max-rounds) cut short, BUT the
                 strategy gathered at least one claim. Data still
                 included in the memo with a caveat.
  - abandoned  : LLM said abandon, OR a cap hit BEFORE any usable
                 claim was extracted.

The synthesizer treats `complete` and `capped` as "include in memo"
and `abandoned` as "outcomes-line only".
"""
from __future__ import annotations

from activegraph import llm_behavior

from ..types import StrategyVerdict
from ._helpers import budget_exhausted, view_objects


MAX_ROUNDS = 3  # max planning rounds per strategy (initial + follow-ups)


def _strategy_has_claims(ctx, sid: str) -> bool:
    return any(
        c.data.get("strategy_id") == sid for c in view_objects(ctx, "claim")
    )


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
    # Scope the LLM's view: only structured objects (strategy / task /
    # claim / source / query / budget_state / overrides). Posts are
    # excluded — they can carry KBs of text each, and the claims already
    # represent the extracted evidence. Without this filter, the default
    # view loads all 18+ posts and overflows Claude's 200K-token cap.
    view={
        "include_types": [
            "strategy",
            "task",
            "claim",
            "source",
            "query",
            "budget_state",
            "overrides",
        ],
        "recent_events": 50,
    },
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
    has_claims = _strategy_has_claims(ctx, sid)
    budget_capped = budget_exhausted(ctx)
    rounds_capped = (
        rounds >= MAX_ROUNDS and llm_output.verdict == "needs_more_tasks"
    )

    # Branch ordering:
    # 1. LLM explicit abandon -- always wins.
    # 2. Budget trip -- capped if we have claims, else abandoned.
    # 3. Max-rounds trip on needs_more -- capped if we have claims, else abandoned.
    # 4. LLM complete -- complete.
    # 5. LLM needs_more + rounds_remaining -- emit follow-up event.
    # 6. Fallback -- complete (LLM noise).

    if llm_output.verdict == "abandon":
        _set_terminal(graph, sid, "abandoned", llm_output.reason or "abandoned")
        graph.emit(
            "strategy.abandoned",
            {"strategy_id": sid, "reason": llm_output.reason or "abandoned"},
        )
        return

    if budget_capped:
        if has_claims:
            _set_terminal(graph, sid, "capped", "budget exhausted")
            graph.emit(
                "strategy.capped",
                {"strategy_id": sid, "reason": "budget exhausted"},
            )
        else:
            _set_terminal(
                graph,
                sid,
                "abandoned",
                "budget exhausted before any usable evidence",
            )
            graph.emit(
                "strategy.abandoned",
                {
                    "strategy_id": sid,
                    "reason": "budget exhausted before any usable evidence",
                },
            )
        return

    if rounds_capped:
        if has_claims:
            _set_terminal(graph, sid, "capped", "max rounds reached")
            graph.emit(
                "strategy.capped",
                {"strategy_id": sid, "reason": "max rounds reached"},
            )
        else:
            _set_terminal(
                graph,
                sid,
                "abandoned",
                "max rounds reached without usable evidence",
            )
            graph.emit(
                "strategy.abandoned",
                {
                    "strategy_id": sid,
                    "reason": "max rounds reached without usable evidence",
                },
            )
        return

    if llm_output.verdict == "complete":
        _set_terminal(graph, sid, "complete", llm_output.feedback)
        graph.emit(
            "strategy.complete",
            {"strategy_id": sid, "summary": llm_output.feedback},
        )
        return

    if llm_output.verdict == "needs_more_tasks":
        # rounds < MAX_ROUNDS (rounds_capped already handled above)
        prior_task_ids = [t.id for t in tasks]
        graph.emit(
            "strategy.needs_more_tasks",
            {
                "strategy_id": sid,
                "feedback": llm_output.feedback,
                "prior_task_ids": prior_task_ids,
            },
        )
        return

    # Fallback: LLM produced a verdict outside the enum somehow.
    _set_terminal(graph, sid, "complete", "completed (fallback)")
    graph.emit(
        "strategy.complete",
        {"strategy_id": sid, "summary": "completed (fallback)"},
    )


def _set_terminal(graph, sid: str, status: str, reason: str) -> None:
    patch = {"status": status}
    if status != "complete":
        patch["abandon_reason"] = reason
    graph.patch_object(sid, patch)
