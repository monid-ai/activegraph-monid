"""L7 synthesizer: combine claims from all strategies into a memo.

Fires on `strategy.complete` and `strategy.abandoned`. Each firing
checks whether EVERY strategy is in a terminal state; the no-op
guard makes all firings except the LAST one cheap. The last firing
writes the memo. Subsequent firings (if any) see the memo and
return.

Why not `runtime.idle`? The runtime emits `runtime.idle` as a
lifecycle event AFTER its dispatch loop exits, so subscribers never
get dispatched. Triggering on strategy termination keeps the
synthesizer in the queue-dispatched event flow.

We declare `view={"recent_events": 2000}` so the cost-summing loop
sees the full LLM event history (the default 50 events would miss
most `llm.responded` events).
"""
from __future__ import annotations

from activegraph import llm_behavior

from ..types import MemoOutput
from ._helpers import view_objects


@llm_behavior(
    name="synthesizer",
    on=["strategy.complete", "strategy.abandoned"],
    description=(
        "You are the FINAL SYNTHESIZER. Every strategy has reached a "
        "terminal state (complete or abandoned). Read the claims grouped "
        "by strategy in the view block and write a 200-300 word memo "
        "that answers the user's goal.\n\n"
        "Requirements:\n"
        "  - Cite each claim inline using its id, e.g. [#claim-12].\n"
        "  - Group findings by strategy where useful.\n"
        "  - Briefly note abandoned strategies (one sentence each) so the "
        "    reader knows what was attempted but didn't pan out.\n"
        "  - Include a `cited_claim_ids` list of every claim id you cited.\n"
        "  - Include a `strategy_outcomes` list with one entry per strategy "
        "    (strategy_id, status, one_line_summary)."
    ),
    output_schema=MemoOutput,
    creates=["memo"],
    view={"recent_events": 2000},
    deterministic=True,
)
def synthesizer(event, graph, ctx, llm_output: MemoOutput):
    # Idempotent: never write more than one memo.
    if view_objects(ctx, "memo"):
        return
    # Wait until every strategy is terminal.
    strategies = view_objects(ctx, "strategy")
    if not strategies:
        return
    if any(s.data.get("status") == "active" for s in strategies):
        return

    monid_cost = sum(
        float(p.data.get("monid_cost", 0.0)) for p in view_objects(ctx, "post")
    )
    llm_cost = _sum_llm_cost_from_events(ctx)

    memo = graph.add_object(
        "memo",
        {
            "summary": llm_output.summary,
            "cited_claim_ids": llm_output.cited_claim_ids,
            "strategy_outcomes": [so.model_dump() for so in llm_output.strategy_outcomes],
            "total_monid_cost": monid_cost,
            "total_llm_cost": llm_cost,
        },
    )
    for cid in llm_output.cited_claim_ids:
        if graph.get_object(cid) is not None:
            graph.add_relation(memo.id, cid, "cited_by")
    for so in llm_output.strategy_outcomes:
        if graph.get_object(so.strategy_id) is not None:
            graph.add_relation(memo.id, so.strategy_id, "summarizes")


def _sum_llm_cost_from_events(ctx) -> float:
    """Sum cost from every `llm.responded` event in the view."""
    total = 0.0
    for e in ctx.view.events(type="llm.responded"):
        cost = e.payload.get("cost_usd") or e.payload.get("cost") or 0.0
        try:
            total += float(cost)
        except (TypeError, ValueError):
            pass
    return total
