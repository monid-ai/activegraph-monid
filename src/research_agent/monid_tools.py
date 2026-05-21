"""activegraph @tool wrappers around the monid HTTP client.

Each tool appears in the activegraph trace as a `tool.requested` /
`tool.responded` event pair, so every external observation is on the
audit trail. The wrappers also benefit from activegraph's tool replay
cache when running under `Runtime(replay_tool_cache=True)`.

We register three tools (discover / inspect / run) instead of one
mega-tool because granular tools yield a readable trace and per-tool
cache hits.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from activegraph.tools import ToolContext, tool

from .config import settings
from .monid_client import MonidClient


# ---- shared client ---------------------------------------------------------

_client = MonidClient(
    api_key=settings.monid_api_key,
    base_url=settings.monid_base_url,
)


# ---- discover -------------------------------------------------------------


class DiscoverInput(BaseModel):
    query: str = Field(description="Natural-language search for monid endpoints.")
    limit: int = Field(10, ge=1, le=20)


class DiscoverCandidate(BaseModel):
    provider: str
    provider_name: str = ""
    endpoint: str
    description: str
    price_type: str = Field("PER_CALL", description="PER_CALL or PER_RESULT")
    price_amount: float = 0.0


class DiscoverOutput(BaseModel):
    candidates: list[DiscoverCandidate]


@tool(
    name="monid_discover",
    description=(
        "Search the monid catalog for data endpoints matching a natural-"
        "language query. Returns ranked candidates with pricing."
    ),
    input_schema=DiscoverInput,
    output_schema=DiscoverOutput,
    cost_per_call=Decimal("0"),
    timeout_seconds=30.0,
    deterministic=False,
)
def monid_discover(args: DiscoverInput, ctx: ToolContext) -> DiscoverOutput:
    raw = _client.discover(query=args.query, limit=args.limit)
    out: list[DiscoverCandidate] = []
    for r in raw.get("results", []):
        price = r.get("price", {}) or {}
        out.append(
            DiscoverCandidate(
                provider=r["provider"],
                provider_name=r.get("providerName", ""),
                endpoint=r["endpoint"],
                description=r.get("description", ""),
                price_type=str(price.get("type", "PER_CALL")),
                price_amount=float(price.get("amount", 0.0)),
            )
        )
    return DiscoverOutput(candidates=out)


# ---- inspect --------------------------------------------------------------


class InspectInput(BaseModel):
    provider: str
    endpoint: str


class InspectOutput(BaseModel):
    provider: str
    endpoint: str
    description: str = ""
    # Keep the schema as a free-form dict; provider input schemas vary
    # too wildly to model strictly here.
    input_schema: dict[str, Any] = Field(default_factory=dict)


@tool(
    name="monid_inspect",
    description=(
        "Inspect a monid endpoint to retrieve its input schema "
        "(pathParams, queryParams, body, bodyType)."
    ),
    input_schema=InspectInput,
    output_schema=InspectOutput,
    cost_per_call=Decimal("0"),
    timeout_seconds=30.0,
    deterministic=False,
)
def monid_inspect(args: InspectInput, ctx: ToolContext) -> InspectOutput:
    raw = _client.inspect(provider=args.provider, endpoint=args.endpoint)
    return InspectOutput(
        provider=args.provider,
        endpoint=args.endpoint,
        description=raw.get("description", ""),
        input_schema=raw.get("input", {}) or {},
    )


# ---- run -----------------------------------------------------------------


class RunInput(BaseModel):
    provider: str
    endpoint: str
    input: dict[str, Any] = Field(default_factory=dict)


class RunOutput(BaseModel):
    run_id: str | None = None
    provider: str
    endpoint: str
    output: Any = None  # list, dict, or None depending on the provider
    cost_usd: float = 0.0
    http_status: int = 200
    status: str = "COMPLETED"


@tool(
    name="monid_run",
    description=(
        "Execute a monid endpoint and return the provider's output. "
        "Handles both sync and async providers transparently (polls "
        "until terminal). Returns the run id, output, and actual cost."
    ),
    input_schema=RunInput,
    output_schema=RunOutput,
    cost_per_call=Decimal("0"),  # actual cost reported in output
    timeout_seconds=150.0,
    deterministic=False,
)
def monid_run(args: RunInput, ctx: ToolContext) -> RunOutput:
    res = _client.run(
        provider=args.provider, endpoint=args.endpoint, input=args.input
    )
    return RunOutput(
        run_id=res.get("run_id"),
        provider=args.provider,
        endpoint=args.endpoint,
        output=res.get("output"),
        cost_usd=float(res.get("cost_usd") or 0.0),
        http_status=int(res.get("http_status") or 200),
        status=res.get("status") or "COMPLETED",
    )


ALL_TOOLS = [monid_discover, monid_inspect, monid_run]
