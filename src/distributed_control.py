"""Distributed control orchestrator: ties the simulation together.

This module runs the closed-loop experiment described in the project brief:

    for each of N control steps (10 kHz loop):
        a) advance PTP synchronisation across all nodes
        b) step the MMC physics
        c) every cell runs **local** fault inference on its own sensor vector
        d) apply the local control decision (voltage balancing + fault response)
        e) log accuracy, sync error and latency

It then contrasts two deployment topologies using a transparent latency model:

    * **Distributed / edge** -- every cell infers locally and in parallel; the
      control-loop critical path is one local inference plus the amortised cost
      of a lightweight PTP sync message. No sensor data leaves the cell.
    * **Centralized** -- every cell ships its sensor vector to a central
      controller over the network, which infers for all cells, then ships the
      decisions back. The critical path is dominated by the network round-trip
      and the serial per-cell inference.

The distributed advantage comes from eliminating the network round-trip from
the real-time control path -- the core argument for edge AI in power
electronics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch

from converter_simulator import MMCSimulator, generate_dataset
from edge_ai import (
    FaultDetector,
    measure_latency,
    model_summary,
    quantize_model,
    train_fault_detector,
)
from synchronization import PTPConfig, PTPNetwork, summarize_sync


@dataclass
class LatencyModel:
    """Parameters of the distributed-vs-centralized latency comparison.

    All times are in milliseconds. Defaults reflect a modest industrial
    Ethernet/EtherCAT fabric: a few tens of microseconds one-way, which is the
    dominant cost a centralized controller pays and a distributed edge node
    avoids.
    """

    network_one_way_ms: float = 0.025   # sensor/command transport, one way
    hub_overhead_ms: float = 0.005      # gather/scatter bookkeeping at the hub
    sync_overhead_ms: float = 0.002     # amortised PTP cost on the edge path


@dataclass
class SimulationResults:
    """Container for everything a single run produces."""

    num_cells: int
    num_steps: int
    dt: float
    bits: int
    model_size: str

    # accuracy / detection
    accuracy_pct: float = 0.0
    detection_latency_ms: float = float("nan")
    predictions: np.ndarray = field(default_factory=lambda: np.empty(0))
    ground_truth: np.ndarray = field(default_factory=lambda: np.empty(0))

    # synchronisation
    sync_error_mean_ns: float = 0.0
    sync_error_max_ns: float = 0.0
    sync_convergence_ms: float = 0.0
    sync_error_history: np.ndarray = field(default_factory=lambda: np.empty(0))
    sync_time: np.ndarray = field(default_factory=lambda: np.empty(0))

    # latency (milliseconds)
    edge_inference_ms: float = 0.0      # one local inference on a single vector
    central_inference_ms: float = 0.0   # one single-vector inference at the hub
    central_batch_ms: float = 0.0       # one batched inference over all cells

    # traces for plotting
    fault_cell: int = -1
    fault_cell_temperature: np.ndarray = field(default_factory=lambda: np.empty(0))
    sensor_stream: Optional[np.ndarray] = None  # (steps, cells, 3)

    # structural
    model_params: int = 0
    model_macs: int = 0


class DistributedController:
    """Orchestrates one distributed edge-AI MMC experiment.

    Parameters
    ----------
    num_cells:
        Number of MMC cells / edge nodes.
    model_size:
        ``"tiny"`` / ``"small"`` / ``"medium"`` detector preset.
    bits:
        Quantisation bit-width deployed on the edge nodes (32/16/8/4).
    sync_interval_ms:
        PTP sync period in milliseconds.
    control_freq_hz:
        Control-loop frequency (default 10 kHz -> dt = 1e-4 s).
    seed:
        Master RNG seed for full reproducibility.
    detector:
        Optional pre-trained :class:`FaultDetector` (skips training).
    """

    def __init__(
        self,
        num_cells: int = 10,
        model_size: str = "small",
        bits: int = 32,
        sync_interval_ms: float = 10.0,
        control_freq_hz: float = 10_000.0,
        seed: int = 0,
        detector: Optional[FaultDetector] = None,
        latency_model: Optional[LatencyModel] = None,
    ) -> None:
        self.num_cells = num_cells
        self.model_size = model_size
        self.bits = bits
        self.sync_interval_ms = sync_interval_ms
        self.dt = 1.0 / control_freq_hz
        self.seed = seed
        self.latency_model = latency_model or LatencyModel()

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Train (or reuse) the shared detector, then deploy a quantised copy on
        # the edge nodes. The full-precision model represents the centralized
        # controller's copy.
        self.train_X, self.train_y = generate_dataset(seed=seed + 1)
        if detector is None:
            detector, _ = train_fault_detector(
                self.train_X, self.train_y, size=model_size, seed=seed
            )
        self.detector = detector
        self.edge_model = quantize_model(detector, bits)
        self.central_model = detector  # fp32 reference at the hub

        self.simulator = MMCSimulator(
            num_cells=num_cells, dt=self.dt, seed=seed + 2
        )
        ptp_cfg = PTPConfig(sync_interval_s=sync_interval_ms * 1e-3)
        self.ptp = PTPNetwork(num_nodes=num_cells, config=ptp_cfg, seed=seed + 3)

    # ------------------------------------------------------------------- loop
    def run(self, num_steps: int = 1000) -> SimulationResults:
        """Execute the closed-loop distributed experiment."""
        sim = self.simulator
        sim.reset()
        self.ptp.reset()

        preds = np.zeros((num_steps, self.num_cells), dtype=np.int64)
        truth = np.zeros((num_steps, self.num_cells), dtype=np.int64)
        sensor_stream = np.zeros((num_steps, self.num_cells, 3), dtype=np.float32)
        fault_temp = np.zeros(num_steps, dtype=np.float32)
        sync_err = []   # per-step worst-case true offset
        sync_t = []

        sync_accum = 0.0
        control = np.full(self.num_cells, 0.5, dtype=np.float32)
        feat_idx = ["voltage", "temperature", "current"]

        with torch.no_grad():
            for k in range(num_steps):
                # (a) PTP synchronisation -- clocks drift by one dt every step
                #     and a PTP correction is applied once per sync interval.
                #     Record every node's residual offset for this step.
                self.ptp.advance(self.dt)
                sync_accum += self.dt
                if sync_accum >= self.ptp.config.sync_interval_s:
                    self.ptp.sync_once()
                    sync_accum = 0.0
                sync_err.append([n.offset_s for n in self.ptp.nodes])
                sync_t.append(k * self.dt)

                # (b) physics
                sensors = sim.step(control)
                feats = np.column_stack([sensors[f] for f in feat_idx]).astype(
                    np.float32
                )
                sensor_stream[k] = feats
                fault_temp[k] = sensors["temperature"][sim.fault_cell]

                # (c) local inference on every cell (each node, its own vector)
                logits = self.edge_model(torch.from_numpy(feats))
                cell_pred = logits.argmax(dim=-1).numpy()
                preds[k] = cell_pred
                truth[k] = sim.ground_truth_fault().astype(np.int64)

                # (d) local control: voltage balancing + protective response.
                v = sensors["voltage"]
                control = np.clip(0.5 + 5e-4 * (sim.params.v_nominal - v), 0.0, 1.0)
                control[cell_pred == 1] = 0.1  # de-rate a cell flagged faulty

        results = SimulationResults(
            num_cells=self.num_cells,
            num_steps=num_steps,
            dt=self.dt,
            bits=self.bits,
            model_size=self.model_size,
            fault_cell=sim.fault_cell,
        )

        # accuracy over the whole run
        results.accuracy_pct = 100.0 * float((preds == truth).mean())
        results.predictions = preds
        results.ground_truth = truth
        results.sensor_stream = sensor_stream
        results.fault_cell_temperature = fault_temp
        steps_to_detect = self._detection_latency(preds, truth)
        results.detection_latency_ms = steps_to_detect * self.dt * 1e3

        # synchronisation metrics (reuse the PTP summariser over the run).
        # sync_history is (steps, nodes) with column 0 = grandmaster.
        sync_history = np.array(sync_err)
        sync_time = np.array(sync_t)
        stats = summarize_sync(
            sync_history, sync_time, self.ptp.config.convergence_threshold_s
        )
        results.sync_error_mean_ns = stats["mean_error_ns"]
        results.sync_error_max_ns = stats["max_error_ns"]
        results.sync_convergence_ms = stats["convergence_time_ms"]
        # store the per-step worst-case slave offset (ns) for plotting
        results.sync_error_history = np.abs(sync_history[:, 1:]).max(axis=1) / 1e-9
        results.sync_time = sync_time

        # inference latency: one local vector on an edge node, one single
        # vector at the hub, and one *batched* pass over all cells at the hub
        # (the fair centralized baseline -- a capable CPU vectorises the cells).
        sample = sensor_stream[0, 0:1]
        batch = sensor_stream[0]  # (num_cells, 3)
        results.edge_inference_ms = measure_latency(self.edge_model, sample)
        results.central_inference_ms = measure_latency(self.central_model, sample)
        results.central_batch_ms = measure_latency(self.central_model, batch)

        summary = model_summary(self.detector)
        results.model_params = summary["parameters"]
        results.model_macs = summary["macs"]
        return results

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _detection_latency(preds: np.ndarray, truth: np.ndarray) -> float:
        """Detector reaction latency, in control steps.

        Measured from the step at which the faulty cell *crosses the fault
        threshold* (ground truth becomes 1) to the first step the local
        detector flags it -- i.e. how fast the model reacts once the fault is
        physically observable, not the slower thermal ramp from injection.
        Returned in **control steps** (caller scales by ``dt``); NaN if the
        fault is never detected.
        """
        # Find the cell that actually faults (first column that ever reads 1).
        faulted_cells = np.where(truth.any(axis=0))[0]
        if faulted_cells.size == 0:
            return float("nan")
        cell = faulted_cells[0]
        # First step at/after onset where prediction matches a true fault.
        active = truth[:, cell].astype(bool)
        detected = preds[:, cell].astype(bool) & active
        idx = np.where(detected)[0]
        if idx.size == 0:
            return float("nan")
        true_onset = np.where(active)[0][0]
        return float((idx[0] - true_onset))  # in steps; caller scales by dt

    def latency_comparison(self, results: SimulationResults) -> Dict[str, float]:
        """Distributed vs centralized total-latency model for the whole run.

        See :class:`LatencyModel` for the parameters. Returns per-call and
        total latencies plus the resulting speedup factor.
        """
        lm = self.latency_model
        steps = results.num_steps

        # Distributed critical path per step: one local inference + amortised
        # PTP sync. All cells run concurrently, so cell count does not add up.
        dist_step_ms = results.edge_inference_ms + lm.sync_overhead_ms
        dist_total_ms = dist_step_ms * steps

        # Centralized critical path per step: a full network round-trip (ship
        # sensors out, ship commands back) plus one batched inference for all
        # cells at the hub. The eliminated round-trip is the edge advantage.
        central_step_ms = (
            2 * lm.network_one_way_ms + results.central_batch_ms + lm.hub_overhead_ms
        )
        central_total_ms = central_step_ms * steps

        speedup = central_total_ms / dist_total_ms if dist_total_ms else float("nan")
        return {
            "distributed_per_step_ms": dist_step_ms,
            "centralized_per_step_ms": central_step_ms,
            "distributed_total_ms": dist_total_ms,
            "centralized_total_ms": central_total_ms,
            "edge_inference_ms": results.edge_inference_ms,
            "central_inference_ms": results.central_inference_ms,
            "central_batch_ms": results.central_batch_ms,
            # individual cost components (ms) so callers/plots can break down
            # the centralized path without re-deriving them.
            "network_roundtrip_ms": 2 * lm.network_one_way_ms,
            "hub_overhead_ms": lm.hub_overhead_ms,
            "sync_overhead_ms": lm.sync_overhead_ms,
            "speedup": speedup,
        }


if __name__ == "__main__":  # pragma: no cover - smoke test
    ctrl = DistributedController(num_cells=10, bits=8)
    res = ctrl.run(num_steps=1000)
    cmp = ctrl.latency_comparison(res)
    print(f"accuracy          : {res.accuracy_pct:.2f}%")
    print(f"detection latency : {res.detection_latency_ms:.1f} ms")
    print(f"mean sync error   : {res.sync_error_mean_ns:.1f} ns")
    print(f"speedup           : {cmp['speedup']:.2f}x")
