"""IEEE 1588 Precision Time Protocol (PTP) simulation for the edge nodes.

Every MMC cell is an independent node with its own free-running oscillator that
has a constant frequency error (drift, in parts-per-million) and an initial
phase offset relative to the grandmaster (node 0). PTP periodically estimates
and corrects each slave's offset using the classic two-way message exchange:

    Sync / Follow_Up   master -> slave   gives  t1 (TX) and t2 (RX)
    Delay_Req / Resp   slave  -> master  gives  t3 (TX) and t4 (RX)

    offset = ((t2 - t1) - (t4 - t3)) / 2
    delay  = ((t2 - t1) + (t4 - t3)) / 2

Timestamps are corrupted by Gaussian capture jitter and a small, fixed path
asymmetry, so even after the servo converges a residual sub-microsecond error
remains -- exactly the regime a distributed control loop must tolerate. A
proportional servo (a software clock servo / PI loop with the integral term
disabled by default) drives the offset toward zero each sync interval.

The module reports, per node and over time, the *true* residual offset, plus
summary metrics: mean error, max error, and convergence time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

NS = 1e-9  # nanoseconds -> seconds


@dataclass
class PTPConfig:
    """Tunable parameters of the PTP simulation."""

    sync_interval_s: float = 10e-3       # time between Sync messages (10 ms)
    path_delay_s: float = 1.2e-6         # nominal master<->slave propagation
    path_asymmetry_s: float = 40e-9      # fixed forward/reverse delay mismatch
    timestamp_jitter_s: float = 250e-9   # 1-sigma timestamp capture jitter
    drift_ppm_std: float = 8.0           # spread of oscillator frequency error
    servo_gain: float = 0.7              # proportional correction gain (0..1]
    # A small integral term nulls the residual drift bias but, without anti-
    # windup, slows settling on the large initial offsets. Default 0 (pure
    # proportional) gives fast, stable convergence with a sub-100 ns drift bias.
    servo_integral: float = 0.0
    init_offset_std_s: float = 50e-6     # spread of initial clock offsets (50 us)
    convergence_threshold_s: float = 1e-6  # "synchronised" = |error| < 1 us


@dataclass
class PTPNode:
    """State of a single PTP slave clock."""

    node_id: int
    offset_s: float          # true offset from master (what we try to null out)
    drift_ppm: float         # constant frequency error
    integral_s: float = 0.0  # accumulated servo integral term


@dataclass
class PTPNetwork:
    """A grandmaster plus ``num_nodes - 1`` PTP slaves.

    Parameters
    ----------
    num_nodes:
        Total node count (node 0 is the grandmaster with zero offset).
    config:
        :class:`PTPConfig` instance; a default is created if omitted.
    seed:
        RNG seed for reproducible jitter / drift / initial offsets.
    """

    num_nodes: int = 10
    config: PTPConfig = field(default_factory=PTPConfig)
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self.reset()

    def reset(self) -> None:
        cfg = self.config
        self.time = 0.0
        self.nodes: List[PTPNode] = []
        for i in range(self.num_nodes):
            if i == 0:  # grandmaster: perfect reference
                self.nodes.append(PTPNode(node_id=0, offset_s=0.0, drift_ppm=0.0))
            else:
                self.nodes.append(
                    PTPNode(
                        node_id=i,
                        offset_s=self._rng.normal(0.0, cfg.init_offset_std_s),
                        drift_ppm=self._rng.normal(0.0, cfg.drift_ppm_std),
                    )
                )
        # History: list of (num_nodes,) arrays of true offset after each cycle.
        self.error_history: List[np.ndarray] = []
        self.time_history: List[float] = []

    # ------------------------------------------------------------- one cycle
    def _exchange_offset_estimate(self, node: PTPNode) -> float:
        """One two-way PTP exchange; returns the *estimated* offset for ``node``."""
        cfg = self.config
        d_fwd = cfg.path_delay_s + cfg.path_asymmetry_s
        d_rev = cfg.path_delay_s - cfg.path_asymmetry_s
        jit = lambda: self._rng.normal(0.0, cfg.timestamp_jitter_s)

        # Master -> slave (Sync). Slave timestamps with its offset clock.
        t1 = self.time + jit()
        t2 = self.time + d_fwd + node.offset_s + jit()
        # Slave -> master (Delay_Req), a short moment later.
        t3 = self.time + node.offset_s + jit()
        t4 = self.time + d_rev + jit()

        return ((t2 - t1) - (t4 - t3)) / 2.0

    def sync_once(self) -> np.ndarray:
        """Run one full PTP sync round across all slaves and apply corrections.

        Returns the per-node *true* offset (in seconds) after correction.
        """
        cfg = self.config
        for node in self.nodes[1:]:
            est = self._exchange_offset_estimate(node)
            node.integral_s += cfg.servo_integral * est
            correction = cfg.servo_gain * est + node.integral_s
            node.offset_s -= correction

        errors = np.array([n.offset_s for n in self.nodes])
        self.error_history.append(errors.copy())
        self.time_history.append(self.time)
        return errors

    def advance(self, duration_s: float) -> None:
        """Let the free-running clocks drift for ``duration_s`` (no correction)."""
        for node in self.nodes[1:]:
            node.offset_s += node.drift_ppm * 1e-6 * duration_s
        self.time += duration_s

    # ----------------------------------------------------------------- drive
    def run(self, num_cycles: int = 200) -> Dict[str, np.ndarray]:
        """Run ``num_cycles`` sync intervals and return histories + metrics.

        Returns
        -------
        dict with keys:
            ``sync_error_history`` -- ``(num_cycles, num_nodes)`` true offsets
            ``time``               -- ``(num_cycles,)`` timestamps [s]
            ``node_offsets``       -- final per-node offsets [s]
            ``mean_error_ns`` / ``max_error_ns`` / ``convergence_time_ms``
        """
        self.reset()
        for _ in range(num_cycles):
            self.sync_once()
            self.advance(self.config.sync_interval_s)

        history = np.array(self.error_history)          # (cycles, nodes)
        times = np.array(self.time_history)
        return {
            "sync_error_history": history,
            "time": times,
            "node_offsets": np.array([n.offset_s for n in self.nodes]),
            **summarize_sync(history, times, self.config.convergence_threshold_s),
        }

    def node_timers(self) -> np.ndarray:
        """Current per-node clock reading (master time + residual offset)."""
        return np.array([self.time + n.offset_s for n in self.nodes])


def summarize_sync(
    history: np.ndarray,
    times: np.ndarray,
    threshold_s: float = 1e-6,
    hold_cycles: int = 5,
) -> Dict[str, float]:
    """Compute steady-state sync metrics from an error history.

    ``history`` is ``(cycles, nodes)``. The grandmaster (column 0) is excluded.
    Steady-state statistics use the second half of the run, after the servo has
    converged.

    Convergence time is the first cycle from which the worst slave stays below
    ``threshold_s`` for at least ``hold_cycles`` *consecutive* cycles. Requiring
    a sustained hold (rather than zero violations for the entire remainder)
    makes the metric robust to isolated steady-state jitter spikes -- a single
    late sample crossing the threshold no longer balloons the reported settling
    time. If the worst slave never sustains the hold, ``convergence_time_ms`` is
    ``NaN`` and ``converged`` is ``False``.
    """
    slave_err = np.abs(history[:, 1:])  # drop grandmaster
    if slave_err.size == 0:
        return {
            "mean_error_ns": 0.0,
            "max_error_ns": 0.0,
            "convergence_time_ms": 0.0,
            "converged": True,
        }

    half = len(history) // 2
    steady = slave_err[half:]
    mean_error_ns = float(steady.mean() / NS)
    max_error_ns = float(steady.max() / NS)

    # Convergence: first cycle starting a run of >= hold_cycles consecutive
    # below-threshold cycles (clamped so short histories still resolve).
    worst_per_cycle = slave_err.max(axis=1)
    below = worst_per_cycle < threshold_s
    n = len(below)
    hold = min(hold_cycles, n)
    converged_idx = None
    for i in range(n - hold + 1):
        if below[i : i + hold].all():
            converged_idx = i
            break

    if converged_idx is None:
        convergence_time_ms = float("nan")
        converged = False
    else:
        convergence_time_ms = float(times[converged_idx] * 1e3)
        converged = True

    return {
        "mean_error_ns": mean_error_ns,
        "max_error_ns": max_error_ns,
        "convergence_time_ms": convergence_time_ms,
        "converged": converged,
    }


if __name__ == "__main__":  # pragma: no cover - smoke test
    net = PTPNetwork(num_nodes=10)
    result = net.run(num_cycles=200)
    print(f"mean sync error : {result['mean_error_ns']:.1f} ns")
    print(f"max sync error  : {result['max_error_ns']:.1f} ns")
    print(f"convergence time: {result['convergence_time_ms']:.1f} ms")
