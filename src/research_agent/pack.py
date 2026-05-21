"""Pack-free assembly.

The activegraph framework offers a `Pack` primitive for declarative
type validation, but Pack-bound behaviors require the
`activegraph.packs` decorator family, not the global `@behavior` /
`@llm_behavior` we used. Rather than maintain two registries, we
follow the pattern in the framework's own examples (babyagi.py,
llm_claim_extraction.py): register behaviors globally and rely on
the Pydantic schemas in `types.py` for input validation at LLM
output time.

This module is kept as the central registration entrypoint so the
CLI imports stay clean (`from .pack import register_pack`).
"""
from __future__ import annotations

from .behaviors import register_all


def register_pack() -> None:
    """Register every behavior + tool in the research agent."""
    register_all()


# Backward-compat name for the CLI -- exposes a no-op object so
# `runtime.load_pack(pack)` calls can be skipped without conditionals.
class _NullPack:
    """Sentinel for places that expect a `pack` symbol."""

    name = "research_agent"


pack = _NullPack()
