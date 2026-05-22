# activegraph-monid

> Auditable cross-source research agent on an event-sourced graph.

Built on [activegraph](https://docs.activegraph.ai/) (event-sourced
reactive graph runtime) and [monid](https://docs.monid.ai/) (unified
gateway to hundreds of data endpoints).

A single user goal flows through eleven reactive behaviors:

```
goal -> decomposer (LLM)         ->  1..N strategies
     -> strategy_planner (LLM)   ->  1..M tasks per strategy
                                     (each task: 1-5 tool-class queries)
     -> discoverer               ->  monid /v1/discover (per query, union results)
     -> route_picker (LLM)       ->  pick one endpoint
     -> inspector                ->  monid /v1/inspect
     -> input_builder (LLM)      ->  build valid params
     -> runner                   ->  monid /v1/run
     -> extractor (LLM)          ->  claims per post
     -> strategy_evaluator (LLM) ->  complete | needs_more | abandon
     -> synthesizer (LLM)        ->  final memo with citations
```

Every step is a typed event on the activegraph trace. The whole run is
forkable: change one source pick, replay everything else from the LLM
and tool caches, and structurally diff the resulting memo.

## Setup

Requires Python `>=3.14` and [`uv`](https://docs.astral.sh/uv/).

```bash
cp .env.example .env       # then fill in ANTHROPIC_API_KEY and MONID_API_KEY
uv sync                    # install deps + register the CLI scripts
```

`MONID_BASE_URL` is optional and only needed if you're pointing at a
non-default monid host.

## Run

```bash
uv run research-agent \
    --topic "What are people saying about AI coding assistants this week?" \
    --max-total-endpoints 6 \
    --budget-monid-usd 1.00
```

A tighter, cheaper run:

```bash
uv run research-agent \
    --topic "Evaluate ElevenLabs' funding history" \
    --max-strategies 2 \
    --max-tasks-per-strategy 2 \
    --memo short \
    --max-cost-usd 1.00 \
    --verbose
```

The trace is persisted to `traces/research-<timestamp>.db` (the
`traces/` directory is created automatically). On completion the CLI
prints the memo, strategy outcomes, cited claims, costs, and the exact
`activegraph inspect …` command to open the trace.

Trimmed example of what success looks like:

```
======================================================================
MEMO
======================================================================
ElevenLabs has raised four disclosed rounds totaling ~$281M, most
recently a $180M Series C in Jan 2025 led by a16z and ICONIQ ...

-- strategy outcomes --
  [complete] funding-history-from-press-coverage
  [complete] funding-history-from-investor-disclosures

-- cited claims --
  [c_03] (conf=0.92, rel=0.95) Series C closed Jan 2025 at $180M ...
  [c_07] (conf=0.88, rel=0.90) Series B closed Jun 2023 at $80M ...

Costs: monid=$0.4120  llm=$0.6840

Graph: 2 strategies, 4 tasks, 38 posts, 11 claims

Trace persisted to: traces/research-20260521-174203.db
Inspect with:       activegraph inspect sqlite:///traces/research-20260521-174203.db
```

## Fork-and-diff

After a baseline run, fork it and swap one source:

```bash
uv run research-agent-fork \
    --db traces/research-<timestamp>.db \
    --parent-run <parent-run-id> \
    --swap apify:/some/endpoint=apify:/other/endpoint
```

`--swap` takes `provider:endpoint=provider:endpoint` pairs and may be
repeated. The replay caches (LLM + tool) make the fork cheap: only the
events downstream of the swap re-execute against live services. The
parent run id is the `run_id` recorded inside the persisted SQLite
trace.

## Layout

```
src/research_agent/
  config.py             # pydantic-settings env loader (fail-fast on missing keys)
  monid_client.py       # pure httpx + on-disk response cache
  monid_tools.py        # @tool wrappers for future LLM tool-use
  types.py              # Pydantic schemas for objects + LLM I/O
  cli.py                # research-agent
  fork_cli.py           # research-agent-fork
  pack.py
  behaviors/
    decomposer.py            # goal      -> strategies
    strategy_planner.py      # strategy  -> tasks
    discoverer.py            # task      -> candidate endpoints (monid)
    route_picker.py          # candidates-> chosen endpoint (LLM)
    inspector.py             # endpoint  -> schema (monid)
    input_builder.py         # schema    -> valid params (LLM)
    runner.py                # params    -> results (monid)
    extractor.py             # results   -> claims (LLM)
    strategy_evaluator.py    # claims    -> verdict (LLM)
    synthesizer.py           # claims    -> cited memo (LLM)
    budget_guard.py          # enforces USD + endpoint-count caps
    watcher.py               # friendly-id assignment + running cost
    audit.py                 # invariants on the trace
```

## Test

```bash
ANTHROPIC_API_KEY=test MONID_API_KEY=test uv run pytest tests/ -v
```

The smoke tests are pure Python and do not call external services. The
fake env vars exist only to satisfy `config.py`'s fail-fast loader at
import time.
