"""Watcher: prints one human-readable line per milestone event.

This is a regular `@behavior` (zero LLM cost) subscribing to the
pipeline's milestone events and `click.echo`-ing a one-line summary
for each. It runs in-band with everything else so the output streams
in real time as the pipeline progresses.

Subscriptions are limited to coarse-grained, low-frequency events;
we deliberately do NOT subscribe to `object.created` (way too chatty)
or any `llm.*` / `tool.*` events (use `--verbose` for those via
`activegraph.configure_logging`).

Friendly ids
------------
Raw activegraph object ids are global and monotonic across all object
types, so the first task gets a higher id than the first strategy
(query + budget_state come before). That looks like off-by-one to
users. We assign per-type sequence numbers ('task#1', 'task#2', ...)
at first sighting and reuse them for the rest of the run. The real
ids stay in the SQLite trace; this is purely a display affordance.
"""
from __future__ import annotations

from typing import Any

import click

from activegraph import behavior


_WATCHED = [
    "strategy.proposed",
    "task.proposed",
    "task.candidates.ready",
    "endpoint.selected",
    "endpoint.inspected",
    "endpoint.input_ready",
    "research.run.completed",
    "research.run.failed",
    "task.results.ready",
    "strategy.complete",
    "strategy.capped",
    "strategy.abandoned",
    "strategy.needs_more_tasks",
    "budget.exhausted",
    "claim.unsupported",
]


def _short(s: str | None, n: int = 60) -> str:
    if not s:
        return ""
    s = str(s).replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "\u2026"


# Per-process friendly-id state. Maps real_id -> "kind#N"; the inner
# dict tracks the next sequence number per kind.
_FRIENDLY: dict[str, str] = {}
_NEXT_SEQ: dict[str, int] = {}

# Running monid cost across [run] events (for the watcher's progress
# line). Reset at the start of each CLI invocation.
_RUNNING_MONID_COST: dict[str, float] = {"total": 0.0}


def _friendly(real_id: Any, kind: str) -> str:
    """Return a stable 'kind#N' alias for a raw object id."""
    if real_id is None:
        return f"{kind}#?"
    key = str(real_id)
    cached = _FRIENDLY.get(key)
    if cached is not None:
        return cached
    n = _NEXT_SEQ.get(kind, 0) + 1
    _NEXT_SEQ[kind] = n
    alias = f"{kind}#{n}"
    _FRIENDLY[key] = alias
    return alias


def reset_friendly() -> None:
    """Wipe the alias state. Call once at the start of each CLI run."""
    _FRIENDLY.clear()
    _NEXT_SEQ.clear()
    _RUNNING_MONID_COST["total"] = 0.0


@behavior(name="watcher", on=_WATCHED)
def watcher(event, graph, ctx):
    t = event.type
    p = event.payload

    if t == "strategy.proposed":
        click.echo(
            f"[strategy] {_friendly(p.get('strategy_id'), 'strategy')} "
            f"'{_short(p.get('name'))}' active"
        )
    elif t == "task.proposed":
        click.echo(
            f"[task]     {_friendly(p.get('task_id'), 'task')} "
            f"{_friendly(p.get('strategy_id'), 'strategy')} "
            f"-> '{_short(p.get('description'))}'"
        )
    elif t == "task.candidates.ready":
        n = len(p.get("candidates") or [])
        click.echo(
            f"[discover] {_friendly(p.get('task_id'), 'task')} -> {n} candidates"
        )
    elif t == "endpoint.selected":
        click.echo(
            f"[select]   {_friendly(p.get('task_id'), 'task')} picked "
            f"{p.get('provider', '?')}{p.get('endpoint', '')}"
        )
    elif t == "endpoint.inspected":
        click.echo(
            f"[inspect]  {_friendly(p.get('task_id'), 'task')} schema fetched"
        )
    elif t == "endpoint.input_ready":
        click.echo(
            f"[input]    {_friendly(p.get('task_id'), 'task')} input ready"
        )
    elif t == "research.run.completed":
        cost = float(p.get("cost", 0.0) or 0.0)
        n = int(p.get("post_count", 0) or 0)
        _RUNNING_MONID_COST["total"] += cost
        running = _RUNNING_MONID_COST["total"]
        click.echo(
            f"[run]      {_friendly(p.get('task_id'), 'task')} "
            f"-> {n} posts (${cost:.4f})  total=${running:.4f}"
        )
    elif t == "research.run.failed":
        click.echo(
            f"[run]      {_friendly(p.get('task_id'), 'task')} FAILED: "
            f"{_short(p.get('error'))}"
        )
    elif t == "task.results.ready" and p.get("abandoned"):
        click.echo(
            f"[task]     {_friendly(p.get('task_id'), 'task')} abandoned: "
            f"{_short(p.get('reason'))}"
        )
    elif t == "strategy.complete":
        click.echo(
            f"[strategy] {_friendly(p.get('strategy_id'), 'strategy')} "
            f"-> complete ({_short(p.get('summary'))})"
        )
    elif t == "strategy.capped":
        click.echo(
            f"[strategy] {_friendly(p.get('strategy_id'), 'strategy')} "
            f"-> capped: {_short(p.get('reason'))} (partial evidence kept)"
        )
    elif t == "strategy.abandoned":
        click.echo(
            f"[strategy] {_friendly(p.get('strategy_id'), 'strategy')} "
            f"-> abandoned: {_short(p.get('reason'))}"
        )
    elif t == "strategy.needs_more_tasks":
        click.echo(
            f"[strategy] {_friendly(p.get('strategy_id'), 'strategy')} "
            f"needs more tasks: {_short(p.get('feedback'))}"
        )
    elif t == "budget.exhausted":
        spent = float(p.get("monid_spent_usd", 0.0))
        limit = float(p.get("limit_usd", 0.0))
        endpoints = p.get("endpoint_count")
        max_e = p.get("limit_endpoints") or 0
        trip = p.get("trip_reason") or (
            "endpoints" if (max_e and endpoints and endpoints >= max_e) else "usd"
        )
        if trip == "endpoints":
            reason = f"endpoint count ({endpoints}/{max_e})"
        else:
            reason = f"USD spend (${spent:.4f}/${limit:.2f})"
        click.echo(
            f"[budget]   EXHAUSTED by {reason}.  "
            f"spent=${spent:.4f}/${limit:.2f}  "
            f"endpoints={endpoints}/{max_e or '\u221e'}"
        )
    elif t == "claim.unsupported":
        click.echo(
            f"[claim]    {_friendly(p.get('claim_id'), 'claim')} "
            f"flagged unsupported"
        )
