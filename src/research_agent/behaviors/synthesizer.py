"""L7 synthesizer: combine claims from all strategies into a memo.

Fires on `strategy.complete`, `strategy.capped`, and `strategy.abandoned`.
Each firing checks whether EVERY strategy is in a terminal state; the
no-op guard makes all firings except the LAST one cheap. The last
firing writes the memo.

`capped` strategies contribute their claims to the memo but the prompt
asks the LLM to flag them with a "partial evidence" caveat.

Built via `make_synthesizer(memo_mode=...)` factory so the `--memo`
CLI flag can bake length guidance into the prompt at registration.

memo modes:
  - short    : 80-120 words, cite 3-5 claims, one paragraph.
  - auto     : LLM decides from the claim/strategy count visible.
  - long     : 400-500 words, sectioned, cite 8-15 claims.
  - detailed : 600-800 words, full analysis, cite 15-30 claims.
"""
from __future__ import annotations

from activegraph import llm_behavior

from ..types import MemoOutput
from ._helpers import view_events, view_objects


_BASE_DESCRIPTION = (
    "You are the FINAL SYNTHESIZER. Every strategy has reached a "
    "terminal state (complete, capped, or abandoned). Read the claims "
    "grouped by strategy in the view block and write a memo that "
    "answers the user's goal.\n\n"
    "Requirements (all modes):\n"
    "  - Cite each claim inline using its id, e.g. [#claim-12].\n"
    "  - For CAPPED strategies: include their findings AND add a brief "
    "    caveat that the evidence is partial (e.g. 'budget exhausted "
    "    before full coverage').\n"
    "  - For ABANDONED strategies: do not cite their claims; note them "
    "    in the strategy_outcomes only.\n"
    "  - Group findings by strategy where useful.\n"
    "  - Include a `cited_claim_ids` list of every claim id you cite.\n"
    "  - Include a `strategy_outcomes` entry per strategy with its "
    "    correct status (complete / capped / abandoned).\n\n"
)


_LENGTH_GUIDANCE = {
    "short": (
        "LENGTH: SHORT. Write 80-120 words in a single paragraph. Cite "
        "only the 3-5 strongest claims. Pick the highest-signal evidence."
    ),
    "auto": (
        "LENGTH: AUTO. Pick the right length from the evidence:\n"
        "  - <=5 claims OR 1 strategy: 80-120 words.\n"
        "  - 6-20 claims: 200-300 words.\n"
        "  - 21-40 claims: 400-500 words with brief sections per strategy.\n"
        "  - >40 claims: 600-800 words with full sections per strategy."
    ),
    "long": (
        "LENGTH: LONG. Write 400-500 words organized in sections (one per "
        "strategy). Cite 8-15 claims. Cover each strategy's headline "
        "findings clearly."
    ),
    "detailed": (
        "LENGTH: DETAILED. Write 600-800 words. Use sections per strategy "
        "with sub-bullets for distinct findings. Cite 15-30 claims. "
        "Include nuance, contradictions, and confidence levels where the "
        "evidence supports them."
    ),
}


def make_synthesizer(memo_mode: str = "auto"):
    """Build the synthesizer behavior with a configured memo length mode."""
    if memo_mode not in _LENGTH_GUIDANCE:
        raise ValueError(
            f"memo_mode must be one of {sorted(_LENGTH_GUIDANCE)}; "
            f"got {memo_mode!r}"
        )
    description = _BASE_DESCRIPTION + _LENGTH_GUIDANCE[memo_mode]

    @llm_behavior(
        name="synthesizer",
        on=["strategy.complete", "strategy.capped", "strategy.abandoned"],
        description=description,
        output_schema=MemoOutput,
        creates=["memo"],
        # Explicit view: structured objects only. Posts ARE included
        # because the cost-summing loop below reads `post.monid_cost`,
        # but they're now small (raw_json was dropped in runner.py).
        # Recent events kept at 2000 so the synthesizer can see the
        # full run timeline for context.
        view={
            "include_types": [
                "strategy",
                "task",
                "claim",
                "source",
                "post",
                "query",
                "memo",
                "budget_state",
            ],
            "recent_events": 2000,
        },
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
            float(p.data.get("monid_cost", 0.0))
            for p in view_objects(ctx, "post")
        )
        llm_cost = _sum_llm_cost(ctx)

        memo = graph.add_object(
            "memo",
            {
                "summary": llm_output.summary,
                "cited_claim_ids": llm_output.cited_claim_ids,
                "strategy_outcomes": [
                    so.model_dump() for so in llm_output.strategy_outcomes
                ],
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

    return synthesizer


def _sum_llm_cost(ctx) -> float:
    """Sum cost from every `llm.responded` event in the view."""
    total = 0.0
    for e in view_events(ctx, "llm.responded"):
        cost = e.payload.get("cost_usd") or e.payload.get("cost") or 0.0
        try:
            total += float(cost)
        except (TypeError, ValueError):
            pass
    return total
