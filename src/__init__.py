"""Distributed Edge-AI simulation for Modular Multilevel Converters.

Public API re-exported for convenience::

    from src import DistributedController, FaultDetector, PTPNetwork, MMCSimulator
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Make the sibling modules importable both as ``src.x`` and as bare ``x``
# (the latter keeps the per-module smoke tests and the notebook simple).
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from converter_simulator import (  # noqa: E402,F401
    MMCParameters,
    MMCSimulator,
    generate_dataset,
)
from edge_ai import (  # noqa: E402,F401
    FaultDetector,
    measure_latency,
    quantize_model,
    train_fault_detector,
)
from synchronization import PTPConfig, PTPNetwork  # noqa: E402,F401
from distributed_control import (  # noqa: E402,F401
    DistributedController,
    SimulationResults,
)

__all__ = [
    "MMCParameters",
    "MMCSimulator",
    "generate_dataset",
    "FaultDetector",
    "measure_latency",
    "quantize_model",
    "train_fault_detector",
    "PTPConfig",
    "PTPNetwork",
    "DistributedController",
    "SimulationResults",
]

__version__ = "1.0.0"
