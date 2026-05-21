"""L4 input_builder: build a valid `input` dict from the inspected schema.

LLM, no tools. Sees the endpoint's input schema in the triggering
event's payload (assembled by `inspector`), translates the task into
the schema's expected parameter names, and emits `endpoint.input_ready`
which `runner` consumes.
"""
from __future__ import annotations

from activegraph import llm_behavior

from ..types import InputSpec


_DESCRIPTION = (
    "You are the PARAM CONSTRUCTOR. The triggering event carries an "
    "endpoint's INPUT SCHEMA (JSON Schema-ish, with pathParams, "
    "queryParams, body, bodyType). Read it carefully and build a VALID "
    "`input` dict that:\n"
    "  - Includes every REQUIRED field from the schema.\n"
    "  - Translates the task description into the schema's actual "
    "    parameter names (searchTerms vs keywords vs query vs urls vs ...). "
    "    Pick the most semantically appropriate ones.\n"
    "  - Caps any volume parameter (maxItems / limit / resultsLimit / "
    "    maxResults) at 5 to control cost.\n"
    "  - Passes only ONE value to array parameters unless absolutely "
    "    necessary (cost multiplies per query for many providers).\n"
    "Return the `input` dict and a one-line construction_rationale."
)


@llm_behavior(
    name="input_builder",
    on=["endpoint.inspected"],
    description=_DESCRIPTION,
    output_schema=InputSpec,
    creates=["input_spec"],
    deterministic=True,
)
def input_builder(event, graph, ctx, llm_output: InputSpec):
    p = event.payload
    graph.add_object(
        "input_spec",
        {
            "source_id": p["source_id"],
            "input": dict(llm_output.input),
            "construction_rationale": llm_output.construction_rationale,
        },
    )
    graph.emit(
        "endpoint.input_ready",
        {
            "source_id": p["source_id"],
            "task_id": p["task_id"],
            "strategy_id": p["strategy_id"],
            "provider": p["provider"],
            "endpoint": p["endpoint"],
            "input": dict(llm_output.input),
        },
    )
