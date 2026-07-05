"""Backwards-compatible shim.

The random-pose "mock" docker was replaced by a real gradient-based rigid-body docking
optimizer -- see `analytical_docking.AnalyticalDockingOracle`. These aliases keep old imports
working; prefer importing from `analytical_docking` directly.
"""
from __future__ import annotations

from futureaffinity.datasources.analytical_docking import (
    AnalyticalDockingOracle,
    analytical_energy,
    toy_pairwise_energy,
)

# historical name
MockDockingSource = AnalyticalDockingOracle

__all__ = ["AnalyticalDockingOracle", "MockDockingSource", "analytical_energy", "toy_pairwise_energy"]
