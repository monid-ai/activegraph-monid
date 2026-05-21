"""Inspector: fetch the input schema for the picked endpoint.

Regular behavior (no LLM). Calls `_client.inspect(...)` and emits
`endpoint.inspected` with the schema embedded in the payload so the
downstream `input_builder` LLM can read it from the triggering-event
section of its user message.
"""
from __future__ import annotations

import json

from activegraph import behavior

from ..monid_tools import _client


@behavior(name="inspector", on=["endpoint.selected"])
def inspector(event, graph, ctx):
    p = event.payload
    try:
        schema = _client.inspect(provider=p["provider"], endpoint=p["endpoint"])
    except Exception as exc:
        graph.patch_object(p["task_id"], {"status": "failed"})
        graph.emit(
            "task.results.ready",
            {
                "task_id": p["task_id"],
                "strategy_id": p["strategy_id"],
                "post_ids": [],
                "post_count": 0,
                "abandoned": True,
                "reason": f"inspect failed: {exc}",
            },
        )
        return
    graph.emit(
        "endpoint.inspected",
        {
            "source_id": p["source_id"],
            "task_id": p["task_id"],
            "strategy_id": p["strategy_id"],
            "provider": p["provider"],
            "endpoint": p["endpoint"],
            "description": p.get("description", ""),
            "input_schema_json": json.dumps(
                schema.get("input", {}), indent=2, default=str
            ),
        },
    )
