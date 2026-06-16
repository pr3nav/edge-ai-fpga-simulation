"""Benchmarks, FPGA resource estimation, plots and the end-to-end entry point.

Running ``python src/benchmarks.py`` (or :func:`main`) executes the full
pipeline:

    1. closed-loop distributed run (accuracy, sync error, detection latency)
    2. distributed-vs-centralized latency comparison
    3. quantization sweep (32/16/8/4 bit): accuracy, latency, FPGA footprint
    4. four publication-quality figures saved to ``results/``
    5. a console summary in the exact format requested, plus ``results/summary.txt``

Accuracy, sync error and all FPGA/structural numbers are exactly reproducible
(fixed seeds); the only non-deterministic outputs are the wall-clock inference
latencies (and the speedup derived from them), which are real host
microbenchmarks and vary a few percent run-to-run. Paths are resolved relative
to this file, so the project is GitHub-ready with no machine-specific paths.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from math import ceil, log2
from pathlib import Path
from typing import Dict, List

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless backend: render straight to files
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

# Allow ``python src/benchmarks.py`` as well as ``from src import benchmarks``.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from converter_simulator import export_dataset_csv  # noqa: E402
from distributed_control import DistributedController  # noqa: E402
from edge_ai import (  # noqa: E402
    SUPPORTED_BITS,
    FaultDetector,
    evaluate_accuracy,
    measure_latency,
    quantize_model,
)
from synchronization import PTPConfig, PTPNetwork  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"


# ============================================================ FPGA resources
@dataclass
class FPGAEstimate:
    """Estimated resource footprint of one deployed cell controller."""

    bits: int
    luts: int
    ffs: int
    dsps: int
    bram18: int
    latency_us: float
    fmax_mhz: int


# Design assumptions for a time-multiplexed MAC datapath on a Zynq-7000.
# Calibrated so the default (``small``, 8-bit) detector lands near the
# published ~2,850 LUT figure for comparable PTP + edge-inference cores.
_MACS_PER_CYCLE = {4: 32, 8: 16, 16: 16, 32: 8}   # low bits pack more per cycle
_FMAX_MHZ = {4: 200, 8: 200, 16: 150, 32: 100}    # fixed-point clocks faster
_PIPELINE_DEPTH = 10
_BASE_LUT = 1410   # PTP timestamping, control FSM, AXI glue (fixed overhead)
_BASE_FF = 1200


def _multiplier_lut(bits: int) -> int:
    """LUT cost of a single ``bits``x``bits`` multiplier (~k * bits**2)."""
    return round(1.1 * bits * bits)


def estimate_fpga_resources(model: FaultDetector, bits: int) -> FPGAEstimate:
    """Estimate the FPGA footprint of deploying ``model`` at ``bits`` precision.

    The model is a documented first-order heuristic (not a synthesis report):
    LUT/FF scale with the number of parallel multipliers and the bit-width,
    DSP usage kicks in only above 8-bit (low-bit MACs map to LUTs), BRAM holds
    the quantised weights, and latency is ``cycles / f_max``. It captures the
    *relative* cost of each quantisation level -- the decision the analysis is
    meant to inform -- and is calibrated to published Zynq-7000 designs.
    """
    params = sum(p.numel() for p in model.net.parameters())
    macs = 0
    import torch.nn as nn

    for m in model.net.modules():
        if isinstance(m, nn.Linear):
            macs += m.in_features * m.out_features

    mpc = _MACS_PER_CYCLE[bits]
    fmax = _FMAX_MHZ[bits]
    acc_width = bits + ceil(log2(macs)) + 1

    luts = _BASE_LUT + mpc * _multiplier_lut(bits) + mpc * acc_width
    ffs = _BASE_FF + mpc * bits * 4
    if bits <= 8:
        dsps = 0                       # LUT-based multipliers save DSP slices
    elif bits <= 18:
        dsps = mpc                     # one DSP48 per multiplier
    else:
        dsps = 3 * mpc                 # fp32 multiply ~3 DSP slices each
    bram18 = ceil(params * bits / (18 * 1024))
    cycles = ceil(macs / mpc) + _PIPELINE_DEPTH
    latency_us = cycles / fmax        # cycles / (MHz) -> microseconds

    return FPGAEstimate(
        bits=bits,
        luts=int(luts),
        ffs=int(ffs),
        dsps=int(dsps),
        bram18=int(bram18),
        latency_us=float(latency_us),
        fmax_mhz=int(fmax),
    )


# ============================================================ quantization
@dataclass
class QuantPoint:
    bits: int
    accuracy_pct: float
    cpu_latency_ms: float
    fpga: FPGAEstimate


def quantization_sweep(
    model: FaultDetector, X: np.ndarray, y: np.ndarray
) -> List[QuantPoint]:
    """Accuracy / latency / FPGA footprint at every supported bit-width."""
    points: List[QuantPoint] = []
    sample = X[:1]
    for bits in SUPPORTED_BITS:  # 32, 16, 8, 4
        q = quantize_model(model, bits)
        points.append(
            QuantPoint(
                bits=bits,
                accuracy_pct=evaluate_accuracy(q, X, y),
                cpu_latency_ms=measure_latency(q, sample),
                fpga=estimate_fpga_resources(model, bits),
            )
        )
    return points


# ================================================================== plotting
def _style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 12})


def plot_sync_convergence(path: Path, seed: int = 0) -> None:
    """Figure 1 -- PTP synchronisation error converging over time."""
    net = PTPNetwork(num_nodes=10, config=PTPConfig(sync_interval_s=10e-3), seed=seed)
    res = net.run(num_cycles=200)
    history = np.abs(res["sync_error_history"]) / 1e-9  # -> ns, (cycles, nodes)
    t_ms = res["time"] * 1e3

    fig, ax = plt.subplots(figsize=(10, 6))
    for node in range(1, history.shape[1]):
        ax.plot(t_ms, history[:, node], alpha=0.55, linewidth=1.3)
    ax.plot(
        t_ms, history[:, 1:].max(axis=1), color="black", linewidth=2.5,
        label="worst-case node",
    )
    ax.axhline(1000, color="crimson", ls="--", lw=1.8, label="1 us target")
    ax.axhline(
        res["mean_error_ns"], color="green", ls=":", lw=1.8,
        label=f"steady mean = {res['mean_error_ns']:.0f} ns",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Absolute sync error (ns)")
    ax.set_title("PTP Synchronisation Error Convergence (10 nodes)")
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_latency_comparison(path: Path, cmp: Dict[str, float], num_cells: int) -> None:
    """Figure 2 -- distributed vs centralized control-loop latency."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    labels = ["Distributed\n(edge)", "Centralized\n(hub)"]
    per_step = [cmp["distributed_per_step_ms"] * 1e3, cmp["centralized_per_step_ms"] * 1e3]
    colors = ["#2a9d8f", "#e76f51"]
    bars = ax1.bar(labels, per_step, color=colors)
    ax1.set_ylabel("Control-loop latency per step (us)")
    ax1.set_title("Per-step critical path")
    for b, v in zip(bars, per_step):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f} us",
                 ha="center", va="bottom", fontsize=11)

    # Latency breakdown: the edge path vs each component of the centralized
    # path. Network round-trip and hub overhead are shown separately so no
    # single bar mislabels what it contains.
    comps = {
        "Edge inference": cmp["edge_inference_ms"] * 1e3,
        "Hub inference\n(batched)": cmp["central_batch_ms"] * 1e3,
        "Network\nround-trip": cmp["network_roundtrip_ms"] * 1e3,
        "Hub\noverhead": cmp["hub_overhead_ms"] * 1e3,
    }
    ax2.bar(list(comps.keys()), list(comps.values()),
            color=["#2a9d8f", "#e76f51", "#f4a261", "#e9c46a"])
    ax2.set_ylabel("Latency (us)")
    ax2.set_title(f"Where the time goes  (speedup = {cmp['speedup']:.1f}x)")
    ax2.tick_params(axis="x", labelsize=9)

    fig.suptitle(f"Distributed vs Centralized AI -- {num_cells}-cell MMC", fontsize=15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_accuracy_vs_quant(path: Path, points: List[QuantPoint]) -> None:
    """Figure 3 -- accuracy and FPGA latency across quantisation levels."""
    bits = [p.bits for p in points]
    acc = [p.accuracy_pct for p in points]
    lat = [p.fpga.latency_us for p in points]
    x = np.arange(len(bits))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = "#264653"
    ax1.plot(x, acc, "o-", color=color, lw=2.5, ms=10, label="Accuracy")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{b}-bit" for b in bits])
    ax1.set_ylabel("Fault-detection accuracy (%)", color=color)
    ax1.set_ylim(min(acc) - 3, 100.5)
    ax1.tick_params(axis="y", labelcolor=color)
    for xi, a in zip(x, acc):
        ax1.text(xi, a + 0.3, f"{a:.1f}%", ha="center", fontsize=10)

    ax2 = ax1.twinx()
    color2 = "#e76f51"
    ax2.plot(x, lat, "s--", color=color2, lw=2.5, ms=10, label="FPGA latency")
    ax2.set_ylabel("Est. FPGA inference latency (us)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.grid(False)

    ax1.set_title("Accuracy vs Quantisation Level")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_resource_usage(path: Path, points: List[QuantPoint]) -> None:
    """Figure 4 -- estimated FPGA resource usage per quantisation level."""
    bits = [p.bits for p in points]
    luts = [p.fpga.luts for p in points]
    ffs = [p.fpga.ffs for p in points]
    dsps = [p.fpga.dsps for p in points]
    x = np.arange(len(bits))
    w = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    # LUTs and FFs share a magnitude, so they bar together; DSP counts (0-24)
    # would be invisible at this scale, so they are annotated as text instead.
    ax1.bar(x - w / 2, luts, w, label="LUTs", color="#2a9d8f")
    ax1.bar(x + w / 2, ffs, w, label="Flip-flops", color="#457b9d")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{b}-bit" for b in bits])
    ax1.set_ylabel("Resource count (per cell)")
    ax1.set_title("FPGA Logic Footprint")
    ax1.legend(fontsize=11, loc="upper right")
    top = max(luts)
    for xi, l, ff, d in zip(x, luts, ffs, dsps):
        ax1.text(xi - w / 2, l, f"{l}", ha="center", va="bottom", fontsize=9)
        ax1.text(xi + w / 2, ff, f"DSP48: {d}", ha="center", va="bottom",
                 fontsize=8, color="#9c6644")
    ax1.set_ylim(0, top * 1.12)

    # LUT utilisation on a Zynq-7020 (53,200 LUTs) for the full N-cell system.
    z7020_luts = 53_200
    util = [100 * l / z7020_luts for l in luts]
    ax2.bar([f"{b}-bit" for b in bits], util, color="#264653")
    ax2.set_ylabel("Zynq-7020 LUT utilisation (%)  [1 cell]")
    ax2.set_title("Single-cell utilisation (xc7z020)")
    for xi, u in enumerate(util):
        ax2.text(xi, u, f"{u:.1f}%", ha="center", va="bottom", fontsize=10)

    fig.suptitle("FPGA Resource Estimates vs Quantisation", fontsize=15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ================================================================== summary
def write_summary(
    path: Path,
    results,
    cmp: Dict[str, float],
    points: List[QuantPoint],
    headline_bits: int,
) -> None:
    """Write the human-readable summary report to ``results/summary.txt``."""
    headline = next(p for p in points if p.bits == headline_bits)
    lines: List[str] = []
    add = lines.append

    add("=" * 64)
    add(" Distributed Edge-AI MMC Simulation -- Summary")
    add("=" * 64)
    add("")
    add(f"Cells / edge nodes        : {results.num_cells}")
    add(f"Control steps             : {results.num_steps} @ "
        f"{1.0 / results.dt / 1e3:.0f} kHz "
        f"({results.num_steps * results.dt * 1e3:.0f} ms simulated)")
    add(f"Edge model                : '{results.model_size}' "
        f"({results.model_params} params, {results.model_macs} MACs)")
    add("")
    add("-- Fault detection --------------------------------------------")
    add(f"Accuracy                  : {results.accuracy_pct:.2f} %")
    add(f"Detection latency         : {results.detection_latency_ms:.2f} ms "
        f"after threshold crossing")
    add("")
    add("-- Synchronisation (PTP) --------------------------------------")
    add(f"Mean sync error           : {results.sync_error_mean_ns:.1f} ns")
    add(f"Max sync error            : {results.sync_error_max_ns:.1f} ns")
    add(f"Convergence time          : {results.sync_convergence_ms:.1f} ms")
    add("")
    add("-- Latency: distributed vs centralized ------------------------")
    add(f"Edge inference / call     : {cmp['edge_inference_ms'] * 1e3:.2f} us")
    add(f"Hub inference (batched)   : {cmp['central_batch_ms'] * 1e3:.2f} us")
    add(f"Distributed / step        : {cmp['distributed_per_step_ms'] * 1e3:.2f} us")
    add(f"Centralized / step        : {cmp['centralized_per_step_ms'] * 1e3:.2f} us")
    add(f"Speedup                   : {cmp['speedup']:.2f} x")
    add("")
    add("-- Quantisation sweep -----------------------------------------")
    add(f"{'bits':>5} {'acc %':>8} {'cpu ms':>9} {'fpga us':>9} "
        f"{'LUT':>7} {'DSP':>6} {'BRAM':>5}")
    for p in points:
        add(f"{p.bits:>5} {p.accuracy_pct:>8.2f} {p.cpu_latency_ms:>9.4f} "
            f"{p.fpga.latency_us:>9.3f} {p.fpga.luts:>7} {p.fpga.dsps:>6} "
            f"{p.fpga.bram18:>5}")
    add("")
    add("-- FPGA deployment (headline config) --------------------------")
    add(f"Target                    : Zynq-7000 (xc7z020), {headline_bits}-bit")
    add(f"LUTs (per cell)           : {headline.fpga.luts}")
    add(f"LUTs (full {results.num_cells}-cell system): "
        f"{headline.fpga.luts * results.num_cells}")
    add(f"DSP48 / FF / BRAM18       : {headline.fpga.dsps} / "
        f"{headline.fpga.ffs} / {headline.fpga.bram18}")
    add(f"Est. inference latency    : {headline.fpga.latency_us:.3f} us "
        f"@ {headline.fpga.fmax_mhz} MHz")
    add("")
    path.write_text("\n".join(lines) + "\n")


# ================================================================== driver
def main(
    num_cells: int = 10,
    num_steps: int = 1000,
    model_size: str = "small",
    headline_bits: int = 8,
    seed: int = 0,
) -> Dict[str, object]:
    """Run the full benchmark suite and emit plots, summary and console report."""
    if headline_bits not in SUPPORTED_BITS:
        raise ValueError(
            f"headline_bits must be one of {SUPPORTED_BITS}, got {headline_bits}"
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _style()

    # Persist the reproducible synthetic dataset alongside the code.
    export_dataset_csv(DATA_DIR / "synthetic_converter_data.csv", seed=seed + 1)

    print("Running distributed closed-loop simulation ...")
    ctrl = DistributedController(
        num_cells=num_cells, model_size=model_size, bits=headline_bits, seed=seed
    )
    results = ctrl.run(num_steps=num_steps)
    cmp = ctrl.latency_comparison(results)

    print("Running quantisation sweep ...")
    points = quantization_sweep(ctrl.detector, ctrl.train_X, ctrl.train_y)

    print("Rendering figures ...")
    plot_sync_convergence(RESULTS_DIR / "1_sync_convergence.png", seed=seed)
    plot_latency_comparison(RESULTS_DIR / "2_latency_comparison.png", cmp, num_cells)
    plot_accuracy_vs_quant(RESULTS_DIR / "3_accuracy_vs_quantization.png", points)
    plot_resource_usage(RESULTS_DIR / "4_resource_usage.png", points)

    write_summary(RESULTS_DIR / "summary.txt", results, cmp, points, headline_bits)

    headline = next(p for p in points if p.bits == headline_bits)

    # --- Console report, in the exact requested format ---------------------
    print("\n" + "=" * 56)
    print(" RESULTS")
    print("=" * 56)
    print(f"Distributed AI speedup: {cmp['speedup']:.1f}x")
    print(f"Mean sync error: {results.sync_error_mean_ns:.0f} ns")
    print(f"Fault detection accuracy: {results.accuracy_pct:.2f}%")
    print(
        f"Zynq-7000 LUT usage estimate: {headline.fpga.luts} "
        f"(from Sync-A paper scale)"
    )
    print("=" * 56)
    print(f"Figures + summary written to: {RESULTS_DIR}")

    return {"results": results, "comparison": cmp, "quant": points}


if __name__ == "__main__":
    main()
