"""Synthetic inventory-domain fixtures shared by the judge-kit tests; no real customer text."""

from __future__ import annotations

import re

from learning_control_plane.judging import MetricVocabulary

INVENTORY_VOCABULARY = MetricVocabulary(
    aliases={
        "units": ("units sold", "units", "unit"),
        "revenue": ("total revenue", "revenue", "sales"),
        "sell_through": ("sell-through rate", "sell through rate", "sell-through", "sell through"),
        "returns": ("returns", "returned units"),
    },
    rate_metrics=frozenset({"sell_through"}),
)
# Store names may end in a season suffix such as "_Q3-2026" that answers drop.
SEASON_SUFFIX = re.compile(r"_?Q[1-4]-\d{4}\s*$")
