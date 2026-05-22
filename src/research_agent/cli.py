"""CLI entrypoint for the research agent.

Usage:
    research-agent --topic "Evaluate ElevenLabs' funding history"
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import click

from activegraph import Frame, Graph, Runtime
from activegraph.llm import AnthropicProvider, RecordingLLMProvider

from .behaviors import (
    DEFAULT_MAX_STRATEGIES,
    DEFAULT_MAX_TASKS_PER_STRATEGY,
    DEFAULT_MEMO_MODE,
    register_all,
)


# Each monid call generates roughly this many activegraph events.
EVENTS_PER_ENDPOINT = 65
EVENTS_FIXED_OVERHEAD = 250
# When max_total_endpoints == 0 (unlimited) and the user does not pass
# --max-events, we use this as the framework safety cap.
EVENTS_UNLIMITED_DEFAULT = 5000


def _auto_event_budget(max_total_endpoints: int) -> int:
    if max_total_endpoints == 0:
        return EVENTS_UNLIMITED_DEFAULT
    return EVENTS_FIXED_OVERHEAD + max_total_endpoints * EVENTS_PER_ENDPOINT


@click.command()
@click.option(
    "--topic",
    required=True,
    help="The user goal -- e.g. 'Evaluate the AI coding assistants market this week'.",
)
@click.option(
    "--max-strategies",
    default=DEFAULT_MAX_STRATEGIES,
    type=int,
    show_default=True,
    help="Hard cap on number of strategies the decomposer may produce.",
)
@click.option(
    "--max-tasks-per-strategy",
    default=DEFAULT_MAX_TASKS_PER_STRATEGY,
    type=int,
    show_default=True,
    help="Hard cap on number of tasks per strategy.",
)
@click.option(
    "--max-total-endpoints",
    default=0,
    type=int,
    show_default=True,
    help=(
        "Hard cap on total monid runs. 0 = unlimited (the USD budget still "
        "applies)."
    ),
)
@click.option(
    "--budget-monid-usd",
    default=1.50,
    type=float,
    show_default=True,
    help="Hard cap on total monid spend (always active).",
)
@click.option(
    "--memo",
    "memo_mode",
    type=click.Choice(["short", "auto", "long", "detailed"]),
    default=DEFAULT_MEMO_MODE,
    show_default=True,
    help=(
        "Memo length. short=80-120w, auto=picks from claim count, "
        "long=400-500w, detailed=600-800w."
    ),
)
@click.option(
    "--max-events",
    default=None,
    type=int,
    help=(
        "Activegraph event budget. Auto-scaled from --max-total-endpoints "
        "when unset (250 + endpoints * 65), or 5000 when endpoints are "
        "unlimited."
    ),
)
@click.option(
    "--max-seconds",
    default=600,
    type=int,
    show_default=True,
    help="Wall-clock budget per run.",
)
@click.option(
    "--max-cost-usd",
    default=3.00,
    type=float,
    show_default=True,
    help="Activegraph LLM cost budget (USD).",
)
@click.option(
    "--db",
    default=None,
    help="Path to the SQLite event store. Defaults to traces/research-<ts>.db.",
)
@click.option(
    "--fixtures-llm",
    default="./fixtures/llm",
    show_default=True,
    help="Directory for RecordingLLMProvider fixtures.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable activegraph's JSON event logger (firehose; for debugging).",
)
def main(
    topic: str,
    max_strategies: int,
    max_tasks_per_strategy: int,
    max_total_endpoints: int,
    budget_monid_usd: float,
    memo_mode: str,
    max_events: int | None,
    max_seconds: int,
    max_cost_usd: float,
    db: str | None,
    fixtures_llm: str,
    verbose: bool,
) -> int:
    if db is None:
        os.makedirs("traces", exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        db = f"traces/research-{stamp}.db"

    if max_events is None:
        max_events = _auto_event_budget(max_total_endpoints)

    if verbose:
        from activegraph import configure_logging

        configure_logging(level="INFO")

    register_all(
        max_strategies=max_strategies,
        max_tasks_per_strategy=max_tasks_per_strategy,
        memo_mode=memo_mode,
    )

    # Reset the watcher's friendly-id table + running cost so each CLI
    # run starts fresh.
    from .behaviors.watcher import reset_friendly

    reset_friendly()

    llm_provider = RecordingLLMProvider(
        AnthropicProvider(),
        fixtures_dir=fixtures_llm,
    )

    graph = Graph()

    # Seed budget state BEFORE running the goal so budget_guard reads
    # the configured limits rather than its defaults.
    graph.add_object(
        "budget_state",
        {
            "monid_spent_usd": 0.0,
            "endpoint_count": 0,
            "exhausted": False,
            "max_total_endpoints": max_total_endpoints,
            "budget_monid_usd": budget_monid_usd,
        },
    )

    endpoint_cap_str = (
        f"\u2264{max_total_endpoints}" if max_total_endpoints > 0 else "unlimited"
    )
    constraints = [
        (
            f"Total monid budget: {endpoint_cap_str} monid runs AND at "
            f"most ${budget_monid_usd:.2f}. Plan strategies x tasks to "
            f"stay within this."
        ),
        (
            f"You may produce at most {max_strategies} strategies and at "
            f"most {max_tasks_per_strategy} tasks per strategy. Use fewer "
            f"when the goal does not require more."
        ),
        "Prefer cheap monid endpoints (PER_CALL under $0.01 when possible).",
        "Cite every claim back to the post it came from.",
        "Stop expanding when evidence is sufficient.",
    ]

    runtime = Runtime(
        graph,
        frame=Frame(goal=topic, constraints=constraints),
        llm_provider=llm_provider,
        persist_to=db,
        budget={
            "max_events": max_events,
            "max_seconds": max_seconds,
            "max_cost_usd": str(max_cost_usd),
        },
    )

    click.echo(
        f"[config]   strategies\u2264{max_strategies} "
        f"tasks/strategy\u2264{max_tasks_per_strategy}  "
        f"monid={endpoint_cap_str} runs / ${budget_monid_usd:.2f}  "
        f"memo={memo_mode}  events\u2264{max_events}"
    )
    runtime.run_goal(topic)
    runtime.save_state()

    _print_results(runtime, db)
    return 0


def _find_budget_trip(g) -> dict | None:
    """Return the payload of the first `budget.exhausted` event, or None."""
    for e in g.events:
        if e.type == "budget.exhausted":
            return e.payload
    return None


def _print_results(runtime: Runtime, db: str) -> None:
    g = runtime.graph
    memos = g.objects(type="memo")
    if memos:
        m = memos[0].data
        click.echo("\n" + "=" * 70)
        click.echo("MEMO")
        click.echo("=" * 70)
        click.echo(m["summary"])
        outcomes = m.get("strategy_outcomes", [])
        if outcomes:
            click.echo("\n-- strategy outcomes --")
            for so in outcomes:
                click.echo(
                    f"  [{so.get('status', '?')}] "
                    f"{so.get('one_line_summary', '')}"
                )

        _print_claims_blocks(g, m.get("cited_claim_ids") or [])

        trip = _find_budget_trip(g)
        if trip is not None:
            trip_reason = trip.get("trip_reason") or "usd"
            if trip_reason == "endpoints":
                summary = (
                    f"endpoint count ({trip.get('endpoint_count')}"
                    f"/{trip.get('limit_endpoints')})"
                )
            else:
                summary = (
                    f"USD spend (${float(trip.get('monid_spent_usd', 0)):.4f}"
                    f"/${float(trip.get('limit_usd', 0)):.2f})"
                )
            click.echo(f"\nBudget trip: {summary}")
            click.echo(
                f"  spend at trip = ${float(trip.get('monid_spent_usd', 0)):.4f}"
            )

        click.echo(
            f"\nCosts: monid=${m.get('total_monid_cost', 0):.4f}  "
            f"llm=${m.get('total_llm_cost', 0):.4f}"
        )
    else:
        click.echo("\n(no memo produced; check the trace)")

    n_strategies = len(g.objects(type="strategy"))
    n_tasks = len(g.objects(type="task"))
    n_posts = len(g.objects(type="post"))
    n_claims = len(g.objects(type="claim"))
    click.echo(
        f"\nGraph: {n_strategies} strategies, {n_tasks} tasks, "
        f"{n_posts} posts, {n_claims} claims"
    )

    # Trace tail -- useful when something stalls and there's no memo.
    tail = list(g.events)[-10:]
    if tail:
        click.echo("\n-- trace tail (last 10 events) --")
        for e in tail:
            click.echo(f"  {e.id}  {e.type}")

    url = db if "://" in db else f"sqlite:///{db}"
    click.echo(f"\nTrace persisted to: {db}")
    click.echo(f"Inspect with:       activegraph inspect {url}")


def _print_claims_blocks(g, cited_ids: list[str]) -> None:
    """Print every cited claim's full text, then any uncited claims."""
    all_claims = g.objects(type="claim")
    if not all_claims:
        return

    by_id = {c.id: c for c in all_claims}
    cited_set = set(cited_ids)

    click.echo("\n-- cited claims --")
    if not cited_ids:
        click.echo("  (none cited)")
    else:
        for cid in cited_ids:
            c = by_id.get(cid)
            if c is None:
                click.echo(f"  [{cid}] (missing -- LLM hallucinated this id)")
                continue
            click.echo(
                f"  [{c.id}] "
                f"(conf={c.data.get('confidence', 0):.2f}, "
                f"rel={c.data.get('topic_relevance', 0):.2f}) "
                f"{c.data.get('text', '').strip()}"
            )

    uncited = [c for c in all_claims if c.id not in cited_set]
    if uncited:
        click.echo(
            f"\n-- uncited claims ({len(uncited)} extracted but not cited) --"
        )
        for c in uncited:
            click.echo(
                f"  [{c.id}] "
                f"(conf={c.data.get('confidence', 0):.2f}, "
                f"rel={c.data.get('topic_relevance', 0):.2f}) "
                f"{c.data.get('text', '').strip()}"
            )


if __name__ == "__main__":
    main()
