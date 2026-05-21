"""Fork-and-diff CLI.

Loads a persisted parent run, forks at the goal event, plants
source-override rules on the fork's graph (selector reads these),
resumes the fork to idle with replay caches enabled, and prints a
structural diff against the parent.
"""
from __future__ import annotations

import click

from activegraph import Runtime
from activegraph.llm import AnthropicProvider, RecordingLLMProvider

from .behaviors import register_all


@click.command()
@click.option("--db", required=True, help="Path to the parent SQLite trace.")
@click.option("--parent-run", required=True, help="Parent run id.")
@click.option(
    "--label",
    default="swap-fork",
    show_default=True,
    help="Fork label (used in the fork's run id).",
)
@click.option(
    "--swap",
    multiple=True,
    help=(
        "Endpoint swap rule: from_provider:from_endpoint=to_provider:to_endpoint. "
        "May be repeated."
    ),
)
@click.option(
    "--fixtures-llm",
    default="./fixtures/llm",
    show_default=True,
)
def main(
    db: str,
    parent_run: str,
    label: str,
    swap: tuple[str, ...],
    fixtures_llm: str,
) -> int:
    register_all()

    # Load the parent run.
    parent = Runtime.load(db, run_id=parent_run)

    # Find the goal.created event -- the natural fork point for source
    # selection experiments.
    goal_evt = next(
        (e for e in parent.graph.events if e.type == "goal.created"), None
    )
    if goal_evt is None:
        click.echo("No goal.created event found in parent run; nothing to fork.")
        return 1

    overrides_map: dict[str, str] = {}
    for s in swap:
        try:
            frm, to = s.split("=", 1)
        except ValueError:
            click.echo(f"Bad --swap spec: {s!r}")
            return 2
        overrides_map[frm.strip()] = to.strip()

    llm_provider = RecordingLLMProvider(
        AnthropicProvider(),
        fixtures_dir=fixtures_llm,
    )

    fork = parent.fork(
        at_event=goal_evt.id,
        label=label,
        replay_llm_cache=True,
        replay_tool_cache=True,
        llm_provider=llm_provider,
    )

    # The selector reads `source_overrides` from an `overrides` object
    # on the graph. Plant it before resuming.
    if overrides_map:
        fork.graph.add_object("overrides", {"source_overrides": overrides_map})

    fork.run_until_idle()
    fork.save_state()

    click.echo(f"Fork created: {fork.run_id}")
    _print_diff(parent, fork)
    click.echo(f"\nInspect:  activegraph inspect {db}")
    return 0


def _print_diff(parent: Runtime, fork: Runtime) -> None:
    diff = parent.diff(fork)
    click.echo("\n=== diff: parent vs fork ===")
    click.echo(f"  shared events:       {len(diff.shared_events)}")
    click.echo(f"  parent-only events:  {len(diff.parent_only_events)}")
    click.echo(f"  fork-only events:    {len(diff.fork_only_events)}")
    click.echo(f"  divergent objects:   {len(diff.divergent_objects)}")
    for obj in diff.divergent_objects[:10]:
        click.echo(f"    - {obj.summary()}")
    if len(diff.divergent_objects) > 10:
        click.echo(f"    ... and {len(diff.divergent_objects) - 10} more")
    click.echo(f"  divergent relations: {len(diff.divergent_relations)}")
    for rel in diff.divergent_relations[:5]:
        click.echo(f"    - {rel.summary()}")


if __name__ == "__main__":
    main()
