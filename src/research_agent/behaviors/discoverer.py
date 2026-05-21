"""R1 discoverer: invoke monid_discover for one task.

Regular (non-LLM) behavior. The task already carries `discover_query`
from L2; the LLM judges the *returned candidates* in L3 (selector).

We call the MonidClient directly here rather than going through an
`@tool` wrapper. activegraph's `@tool` is designed for the LLM tool-
use loop (an LLM behavior emits a tool_call, the runtime dispatches
it). Inside a regular `@behavior`, a direct client call is the right
seam; provenance still lives on each `post` object via `monid_run_id`.

If discover returns zero candidates, we still emit a terminal event
(`task.candidates.empty`) and mark the task `failed` so the strategy
can settle. Otherwise pending tasks would block strategy_evaluator
and synthesizer forever.
"""
from __future__ import annotations

from activegraph import behavior

from ..monid_tools import _client  # shared MonidClient instance
from ._helpers import budget_exhausted


@behavior(name="discoverer", on=["task.proposed"])
def discoverer(event, graph, ctx):
    if budget_exhausted(ctx):
        return
    p = event.payload
    raw = _client.discover(query=p["discover_query"], limit=10)
    candidates: list[dict] = []
    for r in raw.get("results", []):
        price = r.get("price", {}) or {}
        candidates.append(
            {
                "provider": r["provider"],
                "endpoint": r["endpoint"],
                "description": r.get("description", ""),
                "price_type": str(price.get("type", "PER_CALL")),
                "price_amount": float(price.get("amount", 0.0)),
            }
        )

    if not candidates:
        # Settle the task so strategy_evaluator can fire for the parent
        # strategy. Without this, an empty discover would hang the run.
        graph.patch_object(p["task_id"], {"status": "failed"})
        graph.emit(
            "task.results.ready",
            {
                "task_id": p["task_id"],
                "strategy_id": p["strategy_id"],
                "post_ids": [],
                "post_count": 0,
                "abandoned": True,
                "reason": "no monid candidates for discover_query",
            },
        )
        return

    graph.emit(
        "task.candidates.ready",
        {
            "task_id": p["task_id"],
            "strategy_id": p["strategy_id"],
            "discover_query": p["discover_query"],
            "candidates": candidates,
        },
    )
