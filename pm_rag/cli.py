"""CLI for pm-rag."""
from __future__ import annotations

import json

import click

from pm_rag._demo import demo_events, demo_graph, demo_traces
from pm_rag.eval import evaluate, extract_cases
from pm_rag.index import build_index, query


@click.group()
@click.version_option()
def main() -> None:
    """pm-rag - process-aware retrieval."""


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
    "--alpha",
    default=0.15,
    show_default=True,
    type=float,
    help="PPR restart probability.",
)
@click.option(
    "--ks",
    default="1,3,5,10",
    show_default=True,
    help="Comma-separated list of k values to score.",
)
def cmd_eval(alpha: float, ks: str) -> None:
    """Evaluate the bundled demo on next-event localization.

    For each `(prefix, next_event)` pair extracted from the demo
    traces, query the index and check whether any symbol mapped from
    `next_event` appears in the top-k retrieved.
    """
    parsed_ks = [int(x.strip()) for x in ks.split(",") if x.strip()]
    idx = build_index(demo_graph(), demo_events())
    cases = extract_cases(demo_traces())
    score = evaluate(idx, cases, ks=parsed_ks, alpha=alpha)
    click.echo(
        json.dumps(
            {
                "task": "next-event-localization",
                "n": score.n,
                "alpha": alpha,
                "top_k": {str(k): round(v, 4) for k, v in score.top_k.items()},
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
