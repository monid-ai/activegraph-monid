"""R1 discoverer: invoke monid_discover for one task.

Regular (non-LLM) behavior. The task already carries `discover_queries`
from L2; the LLM judges the *returned candidates* in L3 (route_picker).

We call the MonidClient directly here rather than going through an
`@tool` wrapper. activegraph's `@tool` is designed for the LLM tool-
use loop (an LLM behavior emits a tool_call, the runtime dispatches
it). Inside a regular `@behavior`, a direct client call is the right
seam; provenance still lives on each `post` object via `monid_run_id`.

If discover returns zero candidates, we still emit a terminal event
(`task.results.ready` with abandoned=True) and mark the task `failed`
so the strategy can settle. Otherwise pending tasks would block
strategy_evaluator and synthesizer forever.
"""
from __future__ import annotations

from activegraph import behavior

from ..monid_tools import _client  # shared MonidClient instance
from ._helpers import budget_exhausted, view_objects


@behavior(name="discoverer", on=["task.proposed"])
def discoverer(event, graph, ctx):
    if budget_exhausted(ctx):
        return
    p = event.payload
    queries = p["discover_queries"]
    
    # Call /v1/discover once per query with limit=5, then union + dedupe.
    seen_keys: set[tuple[str, str]] = set()
    candidates: list[dict] = []
    
    for query in queries:
        raw = _client.discover(query=query, limit=5)
        for r in raw.get("results", []):
            key = (r["provider"], r["endpoint"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            price = r.get("price", {}) or {}
            candidates.append(
                {
                    "provider": r["provider"],
                    "endpoint": r["endpoint"],
                    "description": r.get("description", ""),
                    "price_type": str(price.get("type", "PER_CALL")),
                    "price_amount": float(price.get("amount", 0.0)),
                    "matched_query": query,
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
                "reason": "no monid candidates for any discover_query",
            },
        )
        return

    # Enrich payload with context for route_picker's LLM.
    query_objs = view_objects(ctx, "query")
    goal = query_objs[0].data["topic"] if query_objs else ""
    
    strat_obj = next(
        (s for s in view_objects(ctx, "strategy") if s.id == p["strategy_id"]),
        None,
    )
    strategy_name = strat_obj.data.get("name", "") if strat_obj else ""
    strategy_rationale = strat_obj.data.get("rationale", "") if strat_obj else ""
    
    task_obj = next(
        (t for t in view_objects(ctx, "task") if t.id == p["task_id"]),
        None,
    )
    task_description = task_obj.data.get("description", "") if task_obj else ""
    
    budget_objs = view_objects(ctx, "budget_state")
    if budget_objs:
        b = budget_objs[0].data
        budget_remaining_usd = b.get("budget_monid_usd", 0.0) - b.get("monid_spent_usd", 0.0)
        endpoints_remaining = (
            b.get("max_total_endpoints", 0) - b.get("endpoint_count", 0)
            if b.get("max_total_endpoints", 0) > 0
            else None
        )
    else:
        budget_remaining_usd = 0.0
        endpoints_remaining = None
    
    # Prior picks: sources already tried in this strategy with their claim counts.
    prior_picks = []
    for src in view_objects(ctx, "source"):
        if src.data.get("task_id") == p["task_id"]:
            continue  # skip the current task's source (not picked yet)
        # Check if this source belongs to a task in the same strategy.
        src_task_id = src.data.get("task_id", "")
        src_task = next((t for t in view_objects(ctx, "task") if t.id == src_task_id), None)
        if src_task and src_task.data.get("strategy_id") == p["strategy_id"]:
            claim_count = sum(
                1 for c in view_objects(ctx, "claim")
                if c.data.get("task_id") == src_task_id
            )
            prior_picks.append({
                "provider": src.data["provider"],
                "endpoint": src.data["endpoint"],
                "claim_count": claim_count,
            })
    
    graph.emit(
        "task.candidates.ready",
        {
            "task_id": p["task_id"],
            "strategy_id": p["strategy_id"],
            "discover_queries": queries,
            "candidates": candidates,
            "goal": goal,
            "strategy_name": strategy_name,
            "strategy_rationale": strategy_rationale,
            "task_description": task_description,
            "budget_remaining_usd": budget_remaining_usd,
            "endpoints_remaining": endpoints_remaining,
            "prior_picks": prior_picks,
        },
    )
