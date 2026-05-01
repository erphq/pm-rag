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
