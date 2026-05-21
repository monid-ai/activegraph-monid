"""Auditable cross-source research agent built on activegraph + monid.

A user goal flows through:
    1. decomposer        -> 1..N strategies
    2. strategy_planner  -> 1..M tasks per strategy
    3. discoverer        -> monid_discover per task
    4. selector          -> pick 1+ endpoints per task
    5. input_constructor -> build a valid input for each endpoint
    6. runner            -> monid_run, materialize posts
    7. extractor         -> LLM-extracted claims per post
    8. strategy_evaluator-> decide complete / needs more / abandon
    9. synthesizer       -> final memo citing claims from all strategies

Every step appears as a typed event in the trace; the run is forkable
and the LLM/tool replay caches make A/B testing cheap.
"""
from .config import settings  # noqa: F401  (load env early for fail-fast)
