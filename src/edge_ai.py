"""Lightweight edge fault-detection model and quantization utilities.

The model is the compact feed-forward classifier specified for the project:

    Linear(3 -> H1) -> ReLU -> Linear(H1 -> H2) -> ReLU -> Linear(H2 -> 2)

with three size presets (``tiny`` / ``small`` / ``medium``) so the demo
notebook can trade accuracy against FPGA footprint. The default ``small``
preset is exactly ``3 -> 64 -> 16 -> 2`` as required.

The module also provides:

* fast training on the synthetic dataset from :mod:`converter_simulator`,
* input normalisation baked into the model as buffers (so deployment only
  needs the raw sensor vector),
* emulated quantization to 32 / 16 / 8 / 4 bit, and
* helpers to measure inference latency and parameter / MAC counts that the
  FPGA resource estimator in :mod:`benchmarks` consumes.

On a CPU the low-bit kernels are *emulated* (weights are quantised, math runs
in fp32) -- the metric that matters for FPGA deployment is the resource /
throughput model in :mod:`benchmarks`, not the host microbenchmark. This is
called out explicitly wherever it matters.
"""

from __future__ import annotations

import copy
import time
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn

# Hidden-layer widths for each model-size preset. ``small`` matches the spec.
SIZE_PRESETS: Dict[str, Tuple[int, int]] = {
    "tiny": (16, 8),
    "small": (64, 16),
    "medium": (128, 32),
}

FEATURE_NAMES = ("voltage", "temperature", "current")


class FaultDetector(nn.Module):
    """Compact MLP fault classifier with built-in input normalisation.

    Parameters
    ----------
    size:
        One of ``SIZE_PRESETS`` (``"tiny"``, ``"small"``, ``"medium"``).
    """

    def __init__(self, size: str = "small") -> None:
        super().__init__()
        if size not in SIZE_PRESETS:
            raise ValueError(f"Unknown size '{size}'. Choose from {list(SIZE_PRESETS)}.")
        self.size = size
        h1, h2 = SIZE_PRESETS[size]
        self.net = nn.Sequential(
            nn.Linear(3, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 2),
        )
        # Normalisation buffers (set by ``fit``); identity until trained.
        self.register_buffer("feat_mean", torch.zeros(3))
        self.register_buffer("feat_std", torch.ones(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.feat_mean) / self.feat_std
        return self.net(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices (0 = healthy, 1 = fault)."""
        return self.forward(x).argmax(dim=-1)


def _to_tensor(x: "np.ndarray | torch.Tensor") -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.from_numpy(np.asarray(x, dtype=np.float32))


def train_fault_detector(
    X: np.ndarray,
    y: np.ndarray,
    size: str = "small",
    epochs: int = 60,
    lr: float = 1e-2,
    seed: int = 0,
    verbose: bool = False,
) -> Tuple[FaultDetector, float]:
    """Train a :class:`FaultDetector` on ``(X, y)``.

    Training is intentionally tiny (a few dozen full-batch Adam steps on a few
    thousand samples) so the whole pipeline runs in well under a second.

    Returns
    -------
    (model, train_accuracy)
    """
    torch.manual_seed(seed)
    model = FaultDetector(size=size)

    Xt = _to_tensor(X)
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64))

    # Freeze normalisation statistics from the training set.
    model.feat_mean.copy_(Xt.mean(dim=0))
    model.feat_std.copy_(Xt.std(dim=0).clamp_min(1e-6))

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        logits = model(Xt)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            acc = (logits.argmax(-1) == yt).float().mean().item()
            print(f"  epoch {epoch:3d}  loss={loss.item():.4f}  acc={acc:.3f}")

    model.eval()
    acc = evaluate_accuracy(model, X, y)
    return model, acc


# --------------------------------------------------------------- quantization
SUPPORTED_BITS = (32, 16, 8, 4)


def _fake_quantize_tensor(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-tensor fake quantisation (quantise then de-quantise).

    Emulates a fixed-point representation with ``bits`` bits: weights are
    rounded to the nearest representable level so accuracy reflects the real
    precision loss, while the returned tensor stays fp32 for host execution.
    """
    if bits >= 32:
        return w
    qmax = 2 ** (bits - 1) - 1            # symmetric signed range
    scale = w.abs().max() / qmax
    if scale == 0:
        return w
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    return q * scale


def quantize_model(model: FaultDetector, bits: int) -> nn.Module:
    """Return a quantised copy of ``model`` for the requested bit-width.

    * ``32`` -- unchanged fp32 reference.
    * ``16`` -- weights cast to fp16 precision (emulated half).
    * ``8``  -- real PyTorch dynamic int8 quantisation when available, else an
      emulated 8-bit fake-quant fallback.
    * ``4``  -- emulated 4-bit fake-quant (no native CPU kernel exists).

    The returned object is always callable as ``module(tensor) -> logits``.
    """
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {bits}")

    if bits == 32:
        return copy.deepcopy(model).eval()

    if bits == 8:
        try:
            qmodel = torch.quantization.quantize_dynamic(
                copy.deepcopy(model).eval(), {nn.Linear}, dtype=torch.qint8
            )
            return qmodel
        except Exception:  # pragma: no cover - platform dependent fallback
            pass  # fall through to emulated fake-quant

    # Emulated path for 16 / 4 bit (and 8-bit fallback): clone and quantise
    # every Linear weight in place.
    qmodel = copy.deepcopy(model).eval()
    with torch.no_grad():
        for module in qmodel.modules():
            if isinstance(module, nn.Linear):
                if bits == 16:
                    module.weight.copy_(module.weight.half().float())
                else:
                    module.weight.copy_(_fake_quantize_tensor(module.weight, bits))
    return qmodel


# ------------------------------------------------------------------- metrics
@torch.no_grad()
def evaluate_accuracy(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    """Classification accuracy in percent."""
    Xt = _to_tensor(X)
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
    preds = model(Xt).argmax(dim=-1)
    return 100.0 * (preds == yt).float().mean().item()


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    sample: "np.ndarray | torch.Tensor | None" = None,
    n_iters: int = 200,
    n_warmup: int = 20,
) -> float:
    """Measure single-sample inference latency in milliseconds.

    A single 3-element sensor vector is pushed through the model ``n_iters``
    times; the median per-call latency is returned (median is robust to OS
    scheduling jitter on a host CPU).
    """
    if sample is None:
        sample = np.zeros((1, 3), dtype=np.float32)
    x = _to_tensor(sample).reshape(-1, 3)  # accepts a single vector or a batch

    for _ in range(n_warmup):
        model(x)

    times = np.empty(n_iters)
    for i in range(n_iters):
        t0 = time.perf_counter()
        model(x)
        times[i] = time.perf_counter() - t0
    return float(np.median(times) * 1e3)  # -> milliseconds


def count_parameters(model: FaultDetector) -> int:
    """Total number of weight + bias parameters in the linear layers."""
    return sum(p.numel() for p in model.net.parameters())


def count_macs(model: FaultDetector) -> int:
    """Multiply-accumulate operations for one forward pass.

    For an MLP this is the sum of ``in_features * out_features`` over the
    linear layers -- the quantity that drives FPGA DSP/LUT usage.
    """
    macs = 0
    for module in model.net.modules():
        if isinstance(module, nn.Linear):
            macs += module.in_features * module.out_features
    return macs


def model_summary(model: FaultDetector) -> Dict[str, int]:
    """Convenience bundle of structural metrics."""
    h1, h2 = SIZE_PRESETS[model.size]
    return {
        "hidden1": h1,
        "hidden2": h2,
        "parameters": count_parameters(model),
        "macs": count_macs(model),
    }


if __name__ == "__main__":  # pragma: no cover - smoke test
    from converter_simulator import generate_dataset

    X, y = generate_dataset()
    model, acc = train_fault_detector(X, y, verbose=True)
    print(f"train accuracy: {acc:.2f}%  summary: {model_summary(model)}")
    for bits in SUPPORTED_BITS:
        q = quantize_model(model, bits)
        print(
            f"  {bits:2d}-bit: acc={evaluate_accuracy(q, X, y):.2f}%  "
            f"latency={measure_latency(q):.4f} ms"
        )
