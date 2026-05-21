# monid-activegraph

An auditable cross-source research agent built on
[activegraph](https://docs.activegraph.ai/) (event-sourced reactive graph
runtime) and [monid](https://docs.monid.ai/) (unified gateway to hundreds
of data endpoints).

A single user goal flows through:

```
goal -> decomposer (LLM)        ->  1..5 strategies
     -> strategy_planner (LLM)  ->  1..3 tasks per strategy
     -> discoverer              ->  monid /v1/discover
     -> selector (LLM)          ->  pick endpoints
     -> inspector               ->  monid /v1/inspect
     -> input_constructor (LLM) ->  build valid params
     -> runner                  ->  monid /v1/run
     -> extractor (LLM)         ->  claims per post
     -> strategy_evaluator (LLM)->  complete | needs_more | abandon
     -> synthesizer (LLM)       ->  final memo with citations
```

Every step is a typed event on the activegraph trace. The whole run is
forkable: change one source pick, replay everything else from the LLM
and tool caches, and structurally diff the resulting memo.

## Setup

`MONID_API_KEY` and `ANTHROPIC_API_KEY` are already in your environment.
The project is editable-installed via `uv sync`.

```bash
uv sync           # install / update deps
```

## Run

```bash
uv run research-agent \
    --topic "What are people saying about AI coding assistants this week?" \
    --max-total-endpoints 6 \
    --budget-monid-usd 1.00
```

The trace is persisted to `traces/research-<timestamp>.db`. Inspect any
event with:

```bash
uv run activegraph inspect traces/research-<timestamp>.db
```

## Fork-and-diff

After a baseline run completes, fork it and swap one source:

```bash
uv run research-agent-fork \
    --db traces/research-<timestamp>.db \
    --parent-run <parent-run-id> \
    --swap apify:/some/endpoint=apify:/other/endpoint
```

The replay caches (LLM + tool) make the fork cheap: only the events
downstream of the swap re-execute against live services.

## Layout

```
src/research_agent/
  config.py             # pydantic-settings env loader (fail-fast on missing keys)
  monid_client.py       # pure httpx + on-disk response cache
  monid_tools.py        # @tool wrappers for future LLM tool-use
  types.py              # Pydantic schemas for objects + LLM I/O
  behaviors/            # eleven reactive behaviors that compose the pipeline
  cli.py                # research-agent
  fork_cli.py           # research-agent-fork
```

## Test

```bash
ANTHROPIC_API_KEY=test MONID_API_KEY=test uv run pytest tests/ -v
```

The smoke tests do not call external services.
