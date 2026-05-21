"""View-aware helpers shared by multiple behaviors.

Inside a behavior body, `graph` is a BehaviorGraph (mutations + by-id
lookups only). Read-side queries go through `ctx.view`, whose
`objects`, `relations`, and `events` attributes are CALLABLE methods
(not lists): `ctx.view.objects(type="strategy")`.
"""
from __future__ import annotations

from typing import Any


def view_objects(ctx: Any, type_name: str | None = None) -> list:
    """List view objects, optionally filtered by activegraph object type."""
    if type_name is None:
        return ctx.view.objects()
    return ctx.view.objects(type=type_name)


def view_events(ctx: Any, type_name: str | None = None) -> list:
    """List view events, optionally filtered by event type."""
    if type_name is None:
        return ctx.view.events()
    return ctx.view.events(type=type_name)


def budget_exhausted(ctx: Any) -> bool:
    """True if a `budget_state` exists in the view and is exhausted."""
    states = view_objects(ctx, "budget_state")
    return bool(states) and bool(states[0].data.get("exhausted", False))
