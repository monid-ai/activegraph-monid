"""Budget guard: cap total monid runs + total monid spend.

Counts are taken at `endpoint.input_ready` (the event immediately
preceding the monid HTTP call inside `runner`). This binds the
budget to the *intent to actually call monid*, not to LLM-level
selections. Selections that don't proceed to `input_ready` (e.g.
because the candidate was hallucinated) don't burn budget.

Cost is summed on `research.run.completed` because that's when the
actual `cost_usd` is known.

When budget trips, we force-fail any pending tasks so that
`strategy_evaluator` can settle the strategies and `synthesizer` can
write a memo. Otherwise pending tasks would block the pipeline from
ever reaching `runtime.idle`.
"""
from __future__ import annotations

from activegraph import behavior

from ._helpers import view_objects


DEFAULT_MAX_TOTAL_ENDPOINTS = 10
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
    max_endpoints = int(fresh.data.get("max_total_endpoints", DEFAULT_MAX_TOTAL_ENDPOINTS))
    budget_usd = float(fresh.data.get("budget_monid_usd", DEFAULT_BUDGET_MONID_USD))

    if (
        int(fresh.data.get("endpoint_count", 0)) >= max_endpoints
        or float(fresh.data.get("monid_spent_usd", 0.0)) >= budget_usd
    ):
        graph.patch_object(state.id, {"exhausted": True})
        graph.emit(
            "budget.exhausted",
            {
                "monid_spent_usd": float(fresh.data.get("monid_spent_usd", 0.0)),
                "endpoint_count": int(fresh.data.get("endpoint_count", 0)),
                "limit_usd": budget_usd,
                "limit_endpoints": max_endpoints,
            },
        )
        # Force-fail any pending tasks so the rest of the pipeline can
        # drain to a memo. Without this, strategy_evaluator would wait
        # forever for tasks that will never run.
        for task in view_objects(ctx, "task"):
            if task.data.get("status") == "pending":
                graph.patch_object(task.id, {"status": "failed"})
                graph.emit(
                    "task.results.ready",
                    {
                        "task_id": task.id,
                        "strategy_id": task.data.get("strategy_id"),
                        "post_ids": [],
                        "post_count": 0,
                        "abandoned": True,
                        "reason": "budget exhausted",
                    },
                )
