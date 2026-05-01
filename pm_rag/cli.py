"""CLI for pm-rag."""
from __future__ import annotations

import json

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


if __name__ == "__main__":
    main()
