"""L2 strategy_planner: strategy -> 1..max_tasks_per_strategy tasks.

Fires on both `strategy.proposed` (initial planning) and
`strategy.needs_more_tasks` (follow-up after the evaluator says the
strategy is incomplete). In the follow-up case, the event payload
carries `feedback` from L5 describing what is still missing; the LLM
sees this through the triggering-event section of its user message.

The output schema is built at registration time with the configured
cap so the LLM cannot return more than `max_tasks_per_strategy` tasks.
"""
from __future__ import annotations

from activegraph import llm_behavior

from ..types import make_task_plan_schema


_PROMPT = (
    "You are the STRATEGY PLANNER. A strategy is a research direction; "
    "your job is to break it into between 1 and {max_tasks} CONCRETE "
    "TASKS, each a specific data-fetching question.\n\n"
    "DECIDE HOW MANY TASKS THE STRATEGY ACTUALLY NEEDS:\n"
    "  - If one well-targeted task answers the strategy, return ONE task.\n"
    "  - Use multiple tasks ONLY when they cover meaningfully different "
    "    angles of the strategy.\n"
    "Use the MINIMUM task count that adequately covers the strategy. The "
    "frame's `constraints` lists the total monid-call budget remaining; "
    "stay within it.\n\n"
    "Each task carries TWO fields:\n"
    "  - description: one-sentence specific question this task answers\n"
    "  - discover_query: a short noun phrase suitable for monid catalog "
    "    search (e.g. 'tweets about X', 'news articles about Y', "
    "    'company funding rounds', 'product reviews'). NO quotes.\n\n"
    "On a follow-up event (strategy.needs_more_tasks), the triggering "
    "event will contain `feedback` describing what is still missing. "
    "Propose tasks that fill that gap, NOT duplicates of prior tasks "
    "(which are also visible in the view block)."
)


def make_strategy_planner(max_tasks_per_strategy: int = 3):
    """Build the strategy planner behavior with a configured cap."""
    schema = make_task_plan_schema(max_tasks_per_strategy)
    prompt = _PROMPT.format(max_tasks=max_tasks_per_strategy)

    @llm_behavior(
        name="strategy_planner",
        on=["strategy.proposed", "strategy.needs_more_tasks"],
        description=prompt,
        output_schema=schema,
        creates=["task"],
        deterministic=True,
    )
    def strategy_planner(event, graph, ctx, llm_output):
        strategy_id = event.payload["strategy_id"]
        strat = graph.get_object(strategy_id)
        if strat is None or strat.data.get("status") != "active":
            return
        round_num = int(strat.data.get("round_count", 0)) + 1
        graph.patch_object(strategy_id, {"round_count": round_num})

        for t in llm_output.tasks:
            task = graph.add_object(
                "task",
                {
                    "strategy_id": strategy_id,
                    "description": t.description,
                    "discover_query": t.discover_query,
                    "status": "pending",
                    "round": round_num,
                },
            )
            graph.add_relation(strategy_id, task.id, "decomposes_to")
            graph.emit(
                "task.proposed",
                {
                    "task_id": task.id,
                    "strategy_id": strategy_id,
                    "description": t.description,
                    "discover_query": t.discover_query,
                },
            )

    return strategy_planner
