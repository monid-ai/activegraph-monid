"""L3 route_picker: pick ONE candidate from the task's discover results.

LLM applies semantic judgment to the candidate list returned by
`discoverer`. No LLM tool use (activegraph 1.0.5.post2's Anthropic
provider has a bug in count_tokens for tool-use conversations); the
schema fetch happens in the regular `inspector` behavior next.

Honours fork-time source overrides via an optional `overrides` object
on the graph.
"""
from __future__ import annotations

from activegraph import llm_behavior

from ..types import RouteChoice
from ._helpers import view_objects


_DESCRIPTION = (
    "You are the ROUTE PICKER. Given the GOAL, STRATEGY, TASK, and a list of "
    "monid catalog candidates, pick EXACTLY ONE (provider, endpoint) that best "
    "serves this specific task within the larger research context.\n\n"
    "You will see in the triggering event:\n"
    "  - goal: the user's original question\n"
    "  - strategy_name + strategy_rationale: what research angle this task supports\n"
    "  - task_description: the specific data-fetching question\n"
    "  - candidates: list of tools, each tagged with `matched_query` (the tool-class "
    "    query that surfaced it), plus provider, endpoint, description, price\n"
    "  - budget_remaining_usd: how much monid budget is left\n"
    "  - endpoints_remaining: how many more endpoint calls are allowed (null = unlimited)\n"
    "  - prior_picks: sources already tried in this strategy (with claim counts)\n\n"
    "Selection criteria, in priority order:\n"
    "  1. SEMANTIC FIT: The candidate's description semantically matches the task. "
    "     Use the goal + strategy context to judge decisiveness — for a narrow "
    "     \"is X part of org Y\" lookup, a person-profile-by-name tool is more decisive "
    "     than a generic enrichment endpoint that requires an email you don't have.\n"
    "  2. MATCHED_QUERY DIVERSITY: If prior picks in this strategy already tried one "
    "     tool class (e.g. 'enrich a person') and got few/zero claims, prefer a "
    "     candidate surfaced by a DIFFERENT matched_query (e.g. 'search linkedin', "
    "     'search twitter') to explore a new angle.\n"
    "  3. BUDGET CALIBRATION: Scale spend to the task's decisiveness within the strategy. "
    "     A last-shot critical lookup may justify $0.02-0.05; a broad sentiment trawl "
    "     should stay under $0.01. Check budget_remaining and endpoints_remaining — if "
    "     this is the last allowed call, spend wisely.\n"
    "  4. PRICE TYPE: Prefer PER_CALL over PER_RESULT for cost predictability, unless "
    "     the PER_RESULT option is significantly cheaper per expected unit.\n"
    "  5. PROVIDER QUALITY: When tied, prefer well-known upstream providers over "
    "     re-exposures (e.g. `exa` over `blockrun.ai/api/v1/exa/...`).\n\n"
    "Provider and endpoint strings MUST be copied EXACTLY from the candidate list. "
    "Return an empty endpoint string ONLY if no candidate is a reasonable match."
)


@llm_behavior(
    name="route_picker",
    on=["task.candidates.ready"],
    description=_DESCRIPTION,
    output_schema=RouteChoice,
    creates=["source"],
    deterministic=True,
)
def route_picker(event, graph, ctx, llm_output: RouteChoice):
    p = event.payload
    if not llm_output.endpoint:
        # LLM declined. Fail the task cleanly so its strategy can settle.
        graph.patch_object(p["task_id"], {"status": "failed"})
        graph.emit(
            "task.results.ready",
            {
                "task_id": p["task_id"],
                "strategy_id": p["strategy_id"],
                "post_ids": [],
                "post_count": 0,
                "abandoned": True,
                "reason": "route_picker declined: no candidate fit",
            },
        )
        return

    cand_map = {(c["provider"], c["endpoint"]): c for c in p["candidates"]}
    key = (llm_output.provider, llm_output.endpoint)
    cand = cand_map.get(key)

    # Honour fork-time source overrides.
    overrides = _read_overrides(ctx)
    swap_target = overrides.get(f"{llm_output.provider}:{llm_output.endpoint}")
    if swap_target:
        sp, se = swap_target.split(":", 1)
        cand = {
            "provider": sp,
            "endpoint": se,
            "description": (cand or {}).get("description", "swapped"),
            "price_type": (cand or {}).get("price_type", "PER_CALL"),
            "price_amount": (cand or {}).get("price_amount", 0.0),
        }
        reason = f"[fork swap] {llm_output.selection_reason}"
    elif cand is None:
        graph.patch_object(p["task_id"], {"status": "failed"})
        graph.emit(
            "task.results.ready",
            {
                "task_id": p["task_id"],
                "strategy_id": p["strategy_id"],
                "post_ids": [],
                "post_count": 0,
                "abandoned": True,
                "reason": (
                    f"hallucinated candidate "
                    f"{llm_output.provider}{llm_output.endpoint}"
                ),
            },
        )
        return
    else:
        reason = llm_output.selection_reason

    src = graph.add_object(
        "source",
        {
            "provider": cand["provider"],
            "endpoint": cand["endpoint"],
            "description": cand.get("description", ""),
            "price_type": cand.get("price_type", "PER_CALL"),
            "price_amount": float(cand.get("price_amount", 0.0)),
            "selection_reason": reason,
            "task_id": p["task_id"],
        },
    )
    graph.add_relation(src.id, p["task_id"], "derived_from_task")
    graph.emit(
        "endpoint.selected",
        {
            "source_id": src.id,
            "task_id": p["task_id"],
            "strategy_id": p["strategy_id"],
            "provider": cand["provider"],
            "endpoint": cand["endpoint"],
            "description": cand.get("description", ""),
        },
    )


def _read_overrides(ctx) -> dict[str, str]:
    objs = view_objects(ctx, "overrides")
    if not objs:
        return {}
    return dict(objs[0].data.get("source_overrides", {}))
