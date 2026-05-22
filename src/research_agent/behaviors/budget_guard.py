"""Budget guard: cap total monid runs + total monid spend.

Counts are taken at `endpoint.input_ready` (the event immediately
preceding the monid HTTP call inside `runner`). This binds the
budget to the *intent to actually call monid*, not to LLM-level
selections.

Cost is summed on `research.run.completed` because that's when the
actual `cost_usd` is known.

When budget trips, we force-fail any task that is `pending` AND has
no parent strategy that's still gathering evidence. Strategies with
in-flight tasks continue; `strategy_evaluator` will mark them
`capped` (partial evidence) or `abandoned` (no evidence) when their
remaining tasks settle.

Sentinel: `max_total_endpoints == 0` means "no count cap" (unlimited
monid calls). The USD cap (`budget_monid_usd`) is always active.
"""
from __future__ import annotations

from activegraph import behavior

from ._helpers import view_objects


DEFAULT_MAX_TOTAL_ENDPOINTS = 0  # 0 = unlimited
DEFAULT_BUDGET_MONID_USD = 1.50


def _ensure_state(ctx, graph):
    states = view_objects(ctx, "budget_state")
    if states:
        return states[0]
    return graph.add_object(
        "budget_state",
        {
            "monid_spent_usd": 0.0,
            "endpoint_count": 0,
            "exhausted": False,
            "max_total_endpoints": DEFAULT_MAX_TOTAL_ENDPOINTS,
            "budget_monid_usd": DEFAULT_BUDGET_MONID_USD,
        },
    )


@behavior(
    name="budget_guard",
    on=["endpoint.input_ready", "research.run.completed"],
)
def budget_guard(event, graph, ctx):
    state = _ensure_state(ctx, graph)
    if state.data.get("exhausted"):
        return

    if event.type == "endpoint.input_ready":
        new_count = int(state.data.get("endpoint_count", 0)) + 1
        graph.patch_object(state.id, {"endpoint_count": new_count})
    else:  # research.run.completed
        delta = float(event.payload.get("cost", 0.0))
        new_spent = float(state.data.get("monid_spent_usd", 0.0)) + delta
        graph.patch_object(state.id, {"monid_spent_usd": new_spent})

    fresh = graph.get_object(state.id)
    max_endpoints = int(
        fresh.data.get("max_total_endpoints", DEFAULT_MAX_TOTAL_ENDPOINTS)
    )
    budget_usd = float(
        fresh.data.get("budget_monid_usd", DEFAULT_BUDGET_MONID_USD)
    )
    endpoint_count = int(fresh.data.get("endpoint_count", 0))
    monid_spent = float(fresh.data.get("monid_spent_usd", 0.0))

    # 0 = "unlimited" for the count check; USD cap is always active.
    count_tripped = max_endpoints > 0 and endpoint_count >= max_endpoints
    cost_tripped = monid_spent >= budget_usd

    if count_tripped or cost_tripped:
        graph.patch_object(state.id, {"exhausted": True})
        graph.emit(
            "budget.exhausted",
            {
                "monid_spent_usd": monid_spent,
                "endpoint_count": endpoint_count,
                "limit_usd": budget_usd,
                "limit_endpoints": max_endpoints,
                "trip_reason": "endpoints" if count_tripped else "usd",
            },
        )
        # Force-fail pending tasks (no source picked yet). Tasks that
        # already have a source / input_spec / in-flight monid run are
        # left to finish naturally; their strategies will become
        # `capped` (if they produce claims) or `abandoned` (otherwise)
        # via strategy_evaluator. This preserves partial evidence.
        for task in view_objects(ctx, "task"):
            if task.data.get("status") != "pending":
                continue
            tid = task.id
            has_source = any(
                s.data.get("task_id") == tid for s in view_objects(ctx, "source")
            )
            if has_source:
                # Source already picked; let the existing pipeline carry
                # this task through to its (cached or live) monid call.
                continue
            graph.patch_object(tid, {"status": "failed"})
            graph.emit(
                "task.results.ready",
                {
                    "task_id": tid,
                    "strategy_id": task.data.get("strategy_id"),
                    "post_ids": [],
                    "post_count": 0,
                    "abandoned": True,
                    "reason": "budget exhausted",
                },
            )
