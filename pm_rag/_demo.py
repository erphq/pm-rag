"""Bundled demo: a small order-fulfillment process with a tiny code graph."""
from __future__ import annotations

from pm_rag.graph import CodeGraph

_NODES = [
    "handlers.checkout.order_received_handler",
    "handlers.payment.payment_pending_handler",
    "handlers.payment.payment_settled_handler",
    "handlers.fulfillment.allocate_inventory",
    "handlers.shipping.ship_order",
    "handlers.refund.refund_initiated",
    "handlers.fraud.fraud_review",
    "handlers.notify.delivery_confirmed",
    "utils.money.format_amount",
    "utils.logging.audit_event",
]

_EDGES: list[tuple[int, int, float]] = [
    (0, 1, 1.0),
    (1, 2, 1.0),
    (1, 6, 0.3),
    (2, 3, 1.0),
    (2, 5, 0.3),
    (3, 4, 1.0),
    (4, 7, 1.0),
    (6, 2, 1.0),
]

_EVENTS = [
    "order_received",
    "payment_pending",
    "payment_settled",
    "allocate_inventory",
    "ship_order",
    "refund_initiated",
    "fraud_review",
    "delivery_confirmed",
]


def demo_graph() -> CodeGraph:
    return CodeGraph(nodes=list(_NODES), edges=list(_EDGES))


def demo_events() -> list[str]:
    return list(_EVENTS)


# Synthetic traces over the demo graph. Mix of happy-path, refund,
# fraud-review, and the "with delivery_confirmed" variant.
_HAPPY = [
    "order_received",
    "payment_pending",
    "payment_settled",
    "allocate_inventory",
    "ship_order",
]
_HAPPY_DELIVERED = [*_HAPPY, "delivery_confirmed"]
_REFUNDED = [
    "order_received",
    "payment_pending",
    "payment_settled",
    "refund_initiated",
]
_FRAUD_HAPPY = [
    "order_received",
    "payment_pending",
    "fraud_review",
    "payment_settled",
    "allocate_inventory",
    "ship_order",
]
_FRAUD_REFUNDED = [
    "order_received",
    "payment_pending",
    "fraud_review",
    "payment_settled",
    "refund_initiated",
]

_TRACES: list[list[str]] = [
    _HAPPY,
    _HAPPY_DELIVERED,
    _REFUNDED,
    _FRAUD_HAPPY,
    _HAPPY_DELIVERED,
    _HAPPY,
    _FRAUD_REFUNDED,
    _HAPPY_DELIVERED,
    _HAPPY,
    _REFUNDED,
]


def demo_traces() -> list[list[str]]:
    """Return the demo synthetic traces (list of activity-name lists)."""
    return [list(t) for t in _TRACES]
