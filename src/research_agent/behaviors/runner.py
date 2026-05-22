"""R2 runner: execute one monid endpoint, materialize posts.

Regular behavior. Calls MonidClient directly. The async/sync seam
(some providers return inline, others require polling) is handled
inside the client. We turn each result item into a `post` object with
full provenance.
"""
from __future__ import annotations

import json

from activegraph import behavior

from ..monid_tools import _client
from ._helpers import budget_exhausted


def _extract_text(item) -> str:
    """Best-effort textual representation of a heterogeneous result item.

    Capped at 2000 chars so the extractor LLM sees a real snippet without
    blowing the prompt budget across many posts.
    """
    if isinstance(item, dict):
        for key in ("text", "content", "snippet", "title", "name", "summary"):
            v = item.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()[:2000]
        # Fallback: serialize a small slice
        return json.dumps(item)[:2000]
    if isinstance(item, str):
        return item.strip()[:2000]
    return str(item)[:2000]


@behavior(name="runner", on=["endpoint.input_ready"])
def runner(event, graph, ctx):
    if budget_exhausted(ctx):
        return
    p = event.payload
    try:
        result = _client.run(
            provider=p["provider"], endpoint=p["endpoint"], input=p["input"]
        )
    except Exception as exc:
        graph.patch_object(p["task_id"], {"status": "failed"})
        graph.emit(
            "research.run.failed",
            {
                "source_id": p["source_id"],
                "task_id": p["task_id"],
                "strategy_id": p["strategy_id"],
                "error": str(exc),
            },
        )
        return

    output = result.get("output")
    if isinstance(output, list):
        items = output
    elif output is None:
        items = []
    else:
        items = [output]

    cost_per = (
        float(result.get("cost_usd", 0.0)) / max(len(items), 1) if items else 0.0
    )
    post_ids: list[str] = []
    for item in items:
        if item is None:
            continue
        url = item.get("url") if isinstance(item, dict) else None
        # Raw JSON is NOT stored on the post object — it bloats LLM prompts
        # by 40-100KB per post (we hit a 200K-token overflow on a 18-post run).
        # The full raw response is recoverable from the monid client's
        # on-disk cache (fixtures/monid/*.json) via monid_run_id if needed.
        post = graph.add_object(
            "post",
            {
                "text": _extract_text(item),
                "url": url,
                "source_id": p["source_id"],
                "task_id": p["task_id"],
                "strategy_id": p["strategy_id"],
                "monid_run_id": result.get("run_id"),
                "monid_cost": cost_per,
            },
        )
        graph.add_relation(post.id, p["source_id"], "fetched_from")
        post_ids.append(post.id)

    graph.patch_object(p["task_id"], {"status": "complete"})
    graph.emit(
        "research.run.completed",
        {
            "source_id": p["source_id"],
            "task_id": p["task_id"],
            "strategy_id": p["strategy_id"],
            "run_id": result.get("run_id"),
            "cost": float(result.get("cost_usd", 0.0)),
            "post_count": len(post_ids),
        },
    )
    graph.emit(
        "task.results.ready",
        {
            "task_id": p["task_id"],
            "strategy_id": p["strategy_id"],
            "post_ids": post_ids,
            "post_count": len(post_ids),
        },
    )
