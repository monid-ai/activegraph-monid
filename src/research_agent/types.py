"""Pydantic schemas for graph objects AND for LLM-behavior outputs.

Object schemas could be attached to a Pack for activegraph-side
validation. We rely on Pydantic at the LLM-output boundary instead;
the object types are still documented here for clarity.

LLM I/O schemas come in two flavors:
  - Static schemas (RouteAndInput, ExtractedClaims, StrategyVerdict,
    MemoOutput): one fixed shape used by exactly one behavior.
  - Schema factories (make_decomposition_schema,
    make_task_plan_schema): the `max_length` of the list field is a
    configuration knob, so we build the schema class at registration
    time with the configured cap baked in. This makes the LLM API
    structurally reject over-cap responses.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# =====================================================================
# Graph object schemas (documentary; not currently attached to a Pack)
# =====================================================================


class QueryObj(BaseModel):
    topic: str
    created_at: str


class StrategyObj(BaseModel):
    name: str
    rationale: str
    expected_kind: str = "general"
    status: Literal["active", "complete", "abandoned"] = "active"
    abandon_reason: str | None = None
    round_count: int = 0


class TaskObj(BaseModel):
    strategy_id: str
    description: str
    discover_query: str
    status: Literal["pending", "complete", "failed"] = "pending"
    round: int = 1


class SourceObj(BaseModel):
    provider: str
    endpoint: str
    description: str = ""
    price_type: str = "PER_CALL"
    price_amount: float = 0.0
    selection_reason: str = ""
    task_id: str


class InputSpecObj(BaseModel):
    source_id: str
    input: dict[str, Any]
    construction_rationale: str = ""


class PostObj(BaseModel):
    text: str
    url: str | None = None
    raw_json: dict[str, Any] = Field(default_factory=dict)
    source_id: str
    task_id: str
    strategy_id: str
    monid_run_id: str | None = None
    monid_cost: float = 0.0


class ClaimObj(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    topic_relevance: float = Field(ge=0.0, le=1.0)
    task_id: str
    strategy_id: str


class MemoObj(BaseModel):
    summary: str
    cited_claim_ids: list[str] = Field(default_factory=list)
    strategy_outcomes: list[dict[str, Any]] = Field(default_factory=list)
    total_monid_cost: float = 0.0
    total_llm_cost: float = 0.0


class BudgetStateObj(BaseModel):
    monid_spent_usd: float = 0.0
    endpoint_count: int = 0
    exhausted: bool = False


# =====================================================================
# LLM behavior output schemas \u2014 static
# =====================================================================


# ---- decomposer pieces (the factory below assembles the wrapping schema) -


class StrategyProposal(BaseModel):
    name: str = Field(description="Short name for the strategy, 2-5 words.")
    rationale: str = Field(
        description=(
            "One-sentence explanation of why this research direction matters "
            "for the user goal."
        )
    )
    expected_kind: str = Field(
        default="general",
        description=(
            "Expected source kind: 'social', 'news', 'forum', 'academic', "
            "'enrichment', 'general'."
        ),
    )


# ---- strategy_planner pieces ---------------------------------------------


class TaskProposal(BaseModel):
    description: str = Field(
        description="One-sentence specific question this task answers."
    )
    discover_query: str = Field(
        description=(
            "Short noun phrase for monid discover, e.g. 'tweets about X' or "
            "'news articles about Y'. Do NOT include quotes or punctuation."
        )
    )


# ---- route_picker -> inspector -> input_builder (3-step pipeline) --------


class RouteChoice(BaseModel):
    """Pick exactly ONE candidate from the task's candidates list."""

    provider: str = Field(
        description=(
            "Provider slug copied EXACTLY from the candidate list "
            "(e.g. 'apify'). Empty string means 'no candidate fits'."
        ),
        default="",
    )
    endpoint: str = Field(
        description=(
            "Endpoint path copied EXACTLY from the candidate list "
            "(e.g. '/apidojo/tweet-scraper'). Empty string means 'no candidate fits'."
        ),
        default="",
    )
    selection_reason: str = Field(
        description="One sentence: why this endpoint is the right pick.",
        default="",
    )


class InputSpec(BaseModel):
    """Valid `input` dict built against an endpoint's schema."""

    input: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "A valid JSON dict matching the endpoint's input schema. "
            "Translate the task into the schema's parameter names "
            "(searchTerms / keywords / query / urls / ...); cap volume "
            "parameters (maxItems / limit / resultsLimit) at 5; pass only "
            "ONE value to array parameters."
        ),
    )
    construction_rationale: str = Field(
        default="",
        description="One-line explanation of the parameter choices.",
    )


# ---- strategy_evaluator --------------------------------------------------


class StrategyVerdict(BaseModel):
    verdict: Literal["complete", "needs_more_tasks", "abandon"]
    feedback: str = Field(
        description=(
            "If needs_more_tasks: what is still missing that follow-up tasks "
            "should target. Otherwise a one-line summary."
        ),
        default="",
    )
    reason: str = Field(
        description="Short justification for the verdict.", default=""
    )


# ---- extractor -----------------------------------------------------------


class ExtractedClaim(BaseModel):
    text: str = Field(description="The claim in one short sentence.")
    confidence: float = Field(ge=0.0, le=1.0)
    topic_relevance: float = Field(
        ge=0.0, le=1.0, description="How relevant the claim is to the goal."
    )


class ExtractedClaims(BaseModel):
    claims: list[ExtractedClaim] = Field(default_factory=list)


# ---- synthesizer ---------------------------------------------------------


class StrategyOutcome(BaseModel):
    strategy_id: str
    status: Literal["complete", "abandoned"]
    one_line_summary: str


class MemoOutput(BaseModel):
    summary: str = Field(
        description=(
            "A 200-300 word memo answering the user goal. Cite claims inline "
            "using [#claim-N] notation."
        )
    )
    cited_claim_ids: list[str] = Field(
        default_factory=list,
        description="IDs of claims the memo cites.",
    )
    strategy_outcomes: list[StrategyOutcome] = Field(
        default_factory=list,
        description="One entry per strategy summarizing its outcome.",
    )


# =====================================================================
# LLM behavior output schemas \u2014 factories (cap-configurable)
# =====================================================================


def make_decomposition_schema(max_strategies: int) -> type[BaseModel]:
    """Build a Decomposition schema with `max_strategies` as the list cap.

    Each call returns a fresh schema class. The LLM is structurally
    prevented from emitting more than `max_strategies` strategies
    because Pydantic rejects an over-cap response at parse time.
    """

    class Decomposition(BaseModel):
        strategies: list[StrategyProposal] = Field(
            min_length=1,
            max_length=max_strategies,
            description=(
                f"Between 1 and {max_strategies} distinct research strategies. "
                f"Use FEWER than {max_strategies} when the goal does not "
                f"require all of them."
            ),
        )

    Decomposition.__name__ = f"Decomposition_max{max_strategies}"
    Decomposition.__qualname__ = Decomposition.__name__
    return Decomposition


def make_task_plan_schema(max_tasks: int) -> type[BaseModel]:
    """Build a TaskPlan schema with `max_tasks` as the list cap."""

    class TaskPlan(BaseModel):
        tasks: list[TaskProposal] = Field(
            min_length=1,
            max_length=max_tasks,
            description=(
                f"Between 1 and {max_tasks} concrete data-fetching tasks. "
                f"Use FEWER than {max_tasks} when one task suffices."
            ),
        )

    TaskPlan.__name__ = f"TaskPlan_max{max_tasks}"
    TaskPlan.__qualname__ = TaskPlan.__name__
    return TaskPlan
