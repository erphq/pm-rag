"""CLI for pm-rag."""
from __future__ import annotations

import json
import sys

import click

from pm_rag._demo import demo_events, demo_graph
from pm_rag.index import build_index, query


@click.group()
@click.version_option()
def main() -> None:
    """pm-rag — process-aware retrieval."""


@main.command(name="query")
@click.option("--trace", required=True, help="comma-separated trace prefix")
@click.option("--k", default=10, show_default=True, type=int)
@click.option("--alpha", default=0.15, show_default=True, type=float)
def cmd_query(trace: str, k: int, alpha: float) -> None:
    """Query the bundled demo graph + events with a trace prefix."""
    prefix = [s.strip() for s in trace.split(",") if s.strip()]
    idx = build_index(demo_graph(), demo_events())
    hits = query(idx, prefix, k=k, alpha=alpha)
    payload = [{"symbol": h.symbol, "score": round(h.score, 6)} for h in hits]
    click.echo(json.dumps(payload, indent=2))


@main.command(name="mapping")
def cmd_mapping() -> None:
    """Print the event→symbol mapping for the bundled demo."""
    idx = build_index(demo_graph(), demo_events())
    payload = {
        ev: [idx.graph.nodes[i] for i in nodes]
        for ev, nodes in idx.mapping.items()
    }
    click.echo(json.dumps(payload, indent=2))


@main.command(name="eval")
@click.option(
    "--trace-file",
    required=True,
    type=click.Path(exists=True),
    help='JSONL file; each line: {"prefix": [...], "truth": "event"}.',
)
@click.option("--k", default=1, show_default=True, type=int, help="Top-k cutoff.")
def cmd_eval(trace_file: str, k: int) -> None:
    """Evaluate retrieval accuracy against ground-truth next events.

    NOTE: not yet implemented — see TODO.md for the v0.4 plan.
    """
    # TODO: load trace_file lines into (prefixes, truth) lists
    # TODO: build_index(demo_graph(), demo_events()) or load a custom index
    # TODO: from pm_rag.eval import evaluate; metrics = evaluate(idx, prefixes, truth, k=k)
    # TODO: click.echo(json.dumps(metrics, indent=2))
    click.echo(
        json.dumps({"error": "eval subcommand not yet implemented — see TODO.md"}),
        err=True,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
