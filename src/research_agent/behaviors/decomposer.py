"""L1 decomposer: goal -> 1..max_strategies research strategies.

The output schema is built at registration time with the configured
cap so the LLM is structurally prevented from over-producing.

The Frame's `constraints` carry the budget context (max monid runs +
USD cap), which the LLM sees on every call. The decomposer's prompt
also tells the LLM to plan for that budget and to prefer fewer
strategies when the goal doesn't need many.
"""
from __future__ import annotations

from activegraph import llm_behavior

from ..types import make_decomposition_schema


_PROMPT = (
    "You are the RESEARCH PLANNER in a multi-strategy investigation system.\n\n"
    "Given the user's goal (in the frame / triggering event), propose "
    "between 1 and {max_strategies} DISTINCT research STRATEGIES that, "
    "together, would best answer the goal. Each strategy is a research "
    "direction (NOT a search query yet).\n\n"
    "DECIDE HOW MANY STRATEGIES THE GOAL ACTUALLY NEEDS:\n"
    "  - Simple lookups (\"price of X\", \"what is Y?\") need ONE strategy.\n"
    "  - Sentiment / news monitoring needs ONE strategy (social + news in one).\n"
    "  - Multi-angle investigations (\"evaluate startup X\") may need 3-5.\n"
    "Use the MINIMUM strategy count that gives a complete answer. The "
    "frame's `constraints` lists the total monid-call budget; each "
    "strategy will spawn 1-3 monid calls, so plan to stay within the "
    "budget.\n\n"
    "Good strategies are MUTUALLY DISTINCT angles on the same goal. For "
    "example, for 'evaluate startup X': strategies might be 'funding "
    "history', 'team background', 'product reviews', 'press coverage', "
    "'competitor positioning'. Avoid near-duplicates."
)


def make_decomposer(max_strategies: int = 5):
    """Build the decomposer behavior with a configured cap."""
    schema = make_decomposition_schema(max_strategies)
    prompt = _PROMPT.format(max_strategies=max_strategies)

    @llm_behavior(
        name="decomposer",
        on=["goal.created"],
        description=prompt,
        output_schema=schema,
        creates=["strategy"],
        deterministic=True,
    )
    def decomposer(event, graph, ctx, llm_output):
        goal_text = event.payload.get("goal", "")
        graph.add_object(
            "query",
            {"topic": goal_text, "created_at": str(event.timestamp)},
        )
        for s in llm_output.strategies:
            strat = graph.add_object(
                "strategy",
                {
                    "name": s.name,
                    "rationale": s.rationale,
                    "expected_kind": s.expected_kind,
                    "status": "active",
                    "abandon_reason": None,
                    "round_count": 0,
                },
            )
            graph.emit(
                "strategy.proposed",
                {
                    "strategy_id": strat.id,
                    "name": s.name,
                    "rationale": s.rationale,
                },
            )

    return decomposer
