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
    "You are the ROUTE PICKER. Given a task description and a list of "
    "monid catalog candidates (with provider, endpoint, description, "
    "and price), pick EXACTLY ONE (provider, endpoint) that best "
    "matches the task. Budget is tight; each pick costs a real monid "
    "call.\n\n"
    "Selection criteria, in order:\n"
    "  1. Description semantically matches the task.\n"
    "  2. Lower price is better; prefer PER_CALL over PER_RESULT for "
    "     predictability unless the latter is much cheaper per unit.\n"
    "  3. Prefer broader / better-documented providers when tied.\n\n"
    "Provider and endpoint strings MUST be copied EXACTLY from the "
    "candidate list. Return an empty endpoint string ONLY if no "
    "candidate is a reasonable match."
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
