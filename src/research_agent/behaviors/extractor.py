"""L6 extractor: post text -> typed claims (LLM behavior).

Fires per newly-created `post` object. Uses a depth-1 view so the
LLM can see the post and its incoming `fetched_from` edge to the
source (useful for context like which provider the text came from).
"""
from __future__ import annotations

from activegraph import llm_behavior

from ..types import ExtractedClaims


@llm_behavior(
    name="extractor",
    on=["object.created"],
    where={"object.type": "post"},
    description=(
        "You are the CLAIM EXTRACTOR. Read the post text in the triggering "
        "event and emit 0..N factual claims about the user's goal.\n\n"
        "For each claim, provide:\n"
        "  - text: one short declarative sentence.\n"
        "  - confidence (0..1): how well the post supports the claim.\n"
        "  - topic_relevance (0..1): how relevant the claim is to the goal.\n\n"
        "Skip opinions and unsupported speculation. Keep claims atomic "
        "(one assertion each). Return an empty list if the post is "
        "off-topic or content-free."
    ),
    output_schema=ExtractedClaims,
    creates=["claim"],
    view={"around": "event.payload.object.id", "depth": 1},
    deterministic=True,
)
def extractor(event, graph, ctx, llm_output: ExtractedClaims):
    post = event.payload["object"]
    post_data = post["data"]
    for ec in llm_output.claims:
        if ec.topic_relevance < 0.4:
            continue  # ignore weakly-relevant noise
        claim = graph.add_object(
            "claim",
            {
                "text": ec.text,
                "confidence": ec.confidence,
                "topic_relevance": ec.topic_relevance,
                "task_id": post_data["task_id"],
                "strategy_id": post_data["strategy_id"],
            },
        )
        graph.add_relation(claim.id, post["id"], "extracted_from")
        graph.add_relation(claim.id, post_data["task_id"], "addresses")
