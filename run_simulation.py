#!/usr/bin/env python3
"""Top-level entry point: run the whole edge-AI MMC pipeline end-to-end.

Usage::

    python run_simulation.py                 # 10 cells, 1000 steps, 8-bit edge
    python run_simulation.py --cells 16      # override defaults

Generates the dataset, runs the distributed simulation, the quantisation
sweep and the FPGA analysis, then writes four figures and a summary to
``results/``. Designed to complete in well under 10 seconds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src/`` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from benchmarks import main  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distributed Edge-AI MMC simulation")
    p.add_argument("--cells", type=int, default=10, help="number of MMC cells")
    p.add_argument("--steps", type=int, default=1000, help="control steps")
    p.add_argument(
        "--model-size",
        choices=["tiny", "small", "medium"],
        default="small",
        help="edge detector size preset",
    )
    p.add_argument(
        "--bits",
        type=int,
        choices=[32, 16, 8, 4],
        default=8,
        help="headline edge-deployment bit-width",
    )
    p.add_argument("--seed", type=int, default=0, help="master RNG seed")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        num_cells=args.cells,
        num_steps=args.steps,
        model_size=args.model_size,
        headline_bits=args.bits,
        seed=args.seed,
    )
