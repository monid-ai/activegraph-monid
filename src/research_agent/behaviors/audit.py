"""Audit: catch strategies that produced zero tasks.

If `strategy_planner` produces no tasks for a strategy (LLM parse
failure, vacuous goal, etc.) the strategy would otherwise hang. We
fire shortly after the strategy is proposed and abandon any strategy
that hasn't spawned at least one task by then.

`activate_after` is event-count, not wall-clock. The strategy_planner
is an LLM behavior: behavior.scheduled, behavior.started,
llm.requested, llm.responded, plus N object.created and emit events
for the tasks themselves. 50 events is a safe over-estimate that
gives the planner room across a busy queue.
"""
from __future__ import annotations

from activegraph import behavior

from ._helpers import view_objects


@behavior(
    name="audit_dead_strategy",
    on=["strategy.proposed"],
    activate_after=50,
)
def audit(event, graph, ctx):
    sid = event.payload["strategy_id"]
    strat = graph.get_object(sid)
    if strat is None or strat.data.get("status") != "active":
        return
    tasks = [
        t for t in view_objects(ctx, "task") if t.data.get("strategy_id") == sid
    ]
    if tasks:
        return
    graph.patch_object(
        sid,
        {"status": "abandoned", "abandon_reason": "planner produced no tasks"},
    )
    graph.emit(
        "strategy.abandoned",
        {"strategy_id": sid, "reason": "planner produced no tasks"},
    )
