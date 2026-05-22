"""Reactive behaviors that compose the research agent.

`register_all()` clears the global activegraph registry, then
registers every behavior \u2014 including the cap-configured decomposer,
strategy_planner, and synthesizer built via factories. Safe to call
repeatedly.

CAREFUL: the `@behavior` and `@llm_behavior` decorators in activegraph
auto-register on definition. If you've already imported the behavior
modules (which we do at the top of this file), every static behavior
is already in the registry. Our job in `register_all()`:

  1. clear_registry() empties everything we had at import time.
  2. importlib.reload(monid_tools) re-fires the @tool decorators.
  3. The factory-built behaviors (decomposer, strategy_planner,
     synthesizer) are rebuilt by calling their factories; their
     @llm_behavior re-fires and auto-registers them.
  4. The module-level behaviors lost their registry slot at step 1, so
     we re-add them with explicit register() calls.
"""
from __future__ import annotations

import importlib

from activegraph import (
    clear_registry,
    clear_tool_registry,
    get_registry,
    register,
)

from . import (
    audit,
    budget_guard,
    discoverer,
    extractor,
    input_builder,
    inspector,
    route_picker,
    runner,
    strategy_evaluator,
    watcher,
)
from .decomposer import make_decomposer
from .strategy_planner import make_strategy_planner
from .synthesizer import make_synthesizer


DEFAULT_MAX_STRATEGIES = 5
DEFAULT_MAX_TASKS_PER_STRATEGY = 3
DEFAULT_MEMO_MODE = "auto"


# Behavior instances whose decorator already self-registered at module
# import time. After clear_registry() these instance references are still
# valid; we just need to register() them back in.
_MODULE_LEVEL_BEHAVIORS = [
    discoverer.discoverer,
    route_picker.route_picker,
    inspector.inspector,
    input_builder.input_builder,
    runner.runner,
    extractor.extractor,
    strategy_evaluator.strategy_evaluator,
    budget_guard.budget_guard,
    audit.audit,
    watcher.watcher,
]


def register_all(
    *,
    max_strategies: int = DEFAULT_MAX_STRATEGIES,
    max_tasks_per_strategy: int = DEFAULT_MAX_TASKS_PER_STRATEGY,
    memo_mode: str = DEFAULT_MEMO_MODE,
) -> list:
    """Clear and re-register every behavior and tool.

    The Pydantic output schemas of `decomposer` and `strategy_planner`
    are rebuilt with the configured caps. The synthesizer's prompt is
    rebuilt with the configured memo length guidance.
    """
    clear_registry()
    clear_tool_registry()

    # Re-fire the @tool decorators so the tool registry repopulates.
    from .. import monid_tools

    importlib.reload(monid_tools)

    # Build the cap-configured behaviors. The @llm_behavior decorator
    # inside each factory auto-registers them; do NOT call register()
    # on them again or they'd fire twice per matching event.
    make_decomposer(max_strategies=max_strategies)
    make_strategy_planner(max_tasks_per_strategy=max_tasks_per_strategy)
    make_synthesizer(memo_mode=memo_mode)

    # Re-register the module-level behaviors whose decorators ran at
    # import time but lost their registry slot during clear_registry().
    for b in _MODULE_LEVEL_BEHAVIORS:
        register(b)

    return list(get_registry())


def behavior_count() -> int:
    """Total behaviors the agent registers (3 factory-built + module-level)."""
    return 3 + len(_MODULE_LEVEL_BEHAVIORS)


# Backward-compat: a list view of module-level behaviors only. The
# factory-built decomposer/strategy_planner/synthesizer are NOT
# instantiated at import time (each instantiation auto-registers and
# would leak). Use `register_all()` to get the live, fully-registered
# list.
ALL_BEHAVIORS = list(_MODULE_LEVEL_BEHAVIORS)
