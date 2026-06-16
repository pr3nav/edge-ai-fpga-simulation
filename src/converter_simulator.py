"""Physics-based simulator for a Modular Multilevel Converter (MMC).

The simulator models an N-cell MMC arm where every cell (sub-module) is an
independent edge node carrying three locally measured sensors:

    * ``voltage``      -- sub-module capacitor voltage      [V]
    * ``current``      -- arm current through the cell       [A]
    * ``temperature``  -- junction/case temperature          [degC]

The dynamics are deliberately lightweight (first-order ODEs integrated with
forward Euler) so that 1000+ control steps run in well under a second, while
still producing the qualitative behaviour an edge fault detector must cope with:
capacitor-voltage ripple, I**2*R self-heating, sensor noise, and an injected
thermal fault on a single cell.

Everything here is plain ``numpy`` -- no framework lock-in -- so the same data
generator can feed the PyTorch model in :mod:`edge_ai` or be exported to CSV for
offline experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Ground-truth fault thresholds. A reading is "faulty" when the *true* (noise
# free) state crosses one of these limits. They are used both to label the
# synthetic training set and to score detector accuracy at run time.
TEMP_FAULT_THRESHOLD_C = 70.0      # over-temperature limit
VOLT_DEVIATION_FRACTION = 0.12     # +-12 % capacitor-voltage deviation


@dataclass
class MMCParameters:
    """Electrical / thermal parameters shared by every cell.

    Values are representative of a medium-voltage MMC sub-module. Currents are
    specified *per cell* so the operating point is independent of cell count;
    the healthy steady-state temperature is ~31 degC under the closed-loop
    controller (insertion index ~0.5, what ``run_simulation.py`` uses) and
    ~46 degC under the standalone smoke test (insertion index held at 1.0),
    both comfortably below the 70 degC fault limit and at nominal voltage.
    """

    v_nominal: float = 1000.0      # nominal capacitor voltage [V]
    capacitance: float = 4.5e-3    # sub-module capacitance [F]
    grid_freq: float = 50.0        # fundamental frequency [Hz]
    i_dc: float = 12.0             # per-cell DC arm current [A]
    i_ac_amplitude: float = 80.0   # AC arm-current amplitude [A]
    r_on: float = 8.0e-3           # conduction resistance [ohm]
    switching_loss: float = 6.0    # fixed switching loss [W]
    t_ambient: float = 25.0        # coolant/ambient temperature [degC]
    r_thermal: float = 0.65        # thermal resistance [degC/W]
    # Thermal capacitance is deliberately small so the fault transient is
    # observable inside the 100 ms (1000-step) simulation window. Real
    # sub-modules have a much larger thermal mass (seconds-long transients);
    # the detection logic is identical -- only the time axis is compressed.
    c_thermal: float = 0.02        # effective thermal capacitance [J/degC]
    v_balance_gain: float = 2000.0  # capacitor-voltage balancing stiffness [1/s]
    sensor_noise_pct: float = 0.01  # 1 % Gaussian sensor noise (1 sigma)


@dataclass
class MMCSimulator:
    """Discrete-time simulator for an ``num_cells``-cell MMC.

    Parameters
    ----------
    num_cells:
        Number of sub-modules / edge nodes (default 10).
    dt:
        Integration / control time-step in seconds. Defaults to ``1e-4`` s,
        i.e. a 10 kHz control loop.
    fault_onset_step:
        Control step at which a thermal fault is injected on ``fault_cell``.
        Defaults to step 500, i.e. 50 ms into the 100 ms (1000-step) window at
        a 10 kHz loop. (The reference brief's loose "t = 500 ms" label predates
        the 10 kHz / 1000-step timing; the fault is injected at 50 ms here.)
        Both the step and the cell are configurable.
    fault_cell:
        Index of the cell that develops the fault.
    fault_severity:
        Multiplier applied to the faulty cell's thermal resistance once the
        fault is active (a blocked cooling channel -> the cell runs hot).
    seed:
        RNG seed for reproducible sensor noise.
    """

    num_cells: int = 10
    dt: float = 1e-4
    fault_onset_step: int = 500
    fault_cell: int = 3
    fault_severity: float = 6.0
    seed: int = 0
    params: MMCParameters = field(default_factory=MMCParameters)

    # ------------------------------------------------------------------ setup
    def __post_init__(self) -> None:
        if self.fault_cell >= self.num_cells:
            self.fault_cell = self.num_cells - 1
        self._rng = np.random.default_rng(self.seed)
        self.reset()

    def reset(self) -> None:
        """Return every cell to its nominal operating point.

        Re-seeds the noise RNG so a reset (e.g. at the start of every
        ``DistributedController.run``) restores the *full* state, including the
        sensor-noise stream -- instance reuse is therefore reproducible.
        """
        p = self.params
        self._rng = np.random.default_rng(self.seed)
        self.step_idx = 0
        self.time = 0.0
        # True (noise-free) physical state, one entry per cell. Temperature
        # starts a few degrees above ambient (in the healthy band) and relaxes
        # to its true operating point within the first few ms.
        self.v_cap = np.full(self.num_cells, p.v_nominal, dtype=float)
        self.temperature = np.full(
            self.num_cells, p.t_ambient + 10.0, dtype=float
        )
        self.i_cell = np.zeros(self.num_cells, dtype=float)
        # Per-cell insertion index (control input), 1.0 = fully inserted.
        self.control = np.ones(self.num_cells, dtype=float)
        self._fault_active = False

    # --------------------------------------------------------------- controls
    def apply_control(self, cell_id: int, control_signal: float) -> None:
        """Set the insertion index (in ``[0, 1]``) for a single cell."""
        self.control[cell_id] = float(np.clip(control_signal, 0.0, 1.0))

    def _arm_current(self) -> float:
        """Instantaneous arm current shared by every cell at the current time."""
        p = self.params
        ac = p.i_ac_amplitude * np.sin(2.0 * np.pi * p.grid_freq * self.time)
        return p.i_dc + ac  # per-cell DC + shared AC; independent of cell count

    # ------------------------------------------------------------------- step
    def step(self, control_signals: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Advance the simulation by one ``dt`` and return noisy sensor data.

        Parameters
        ----------
        control_signals:
            Optional array of ``num_cells`` insertion indices. When omitted the
            previously applied control is held.

        Returns
        -------
        dict
            ``{"voltage", "current", "temperature"}`` arrays of length
            ``num_cells`` with Gaussian measurement noise applied.
        """
        p = self.params
        if control_signals is not None:
            self.control = np.clip(np.asarray(control_signals, dtype=float), 0.0, 1.0)

        # Activate the cooling fault once we cross the onset step.
        if self.step_idx >= self.fault_onset_step:
            self._fault_active = True

        i_arm = self._arm_current()
        self.i_cell = self.control * i_arm

        # --- Capacitor voltage: dV/dt = (S * i_arm) / C, with a voltage-
        #     balancing term that pulls the mean back toward nominal. The
        #     balancing stiffness keeps a healthy cell within ~1 % ripple while
        #     leaving room for genuine voltage-deviation faults.
        dv = (self.control * i_arm) / p.capacitance
        balance = -p.v_balance_gain * (self.v_cap - p.v_nominal)
        self.v_cap = self.v_cap + (dv + balance) * self.dt

        # --- Thermal: first-order RC network driven by conduction + switching
        #     losses. The faulty cell loses cooling capacity (higher R_th) and
        #     absorbs extra heat, so its temperature ramps away.
        p_loss = self.i_cell ** 2 * p.r_on + p.switching_loss * self.control
        r_th = np.full(self.num_cells, p.r_thermal)
        if self._fault_active:
            r_th[self.fault_cell] *= self.fault_severity
            p_loss[self.fault_cell] += 30.0  # extra dissipation from the fault
        cooling = (self.temperature - p.t_ambient) / r_th
        self.temperature = self.temperature + (p_loss - cooling) / p.c_thermal * self.dt

        self.step_idx += 1
        self.time += self.dt
        return self.get_sensor_data()

    # --------------------------------------------------------------- readouts
    def _add_noise(self, values: np.ndarray) -> np.ndarray:
        noise = self._rng.normal(0.0, self.params.sensor_noise_pct, size=values.shape)
        return values * (1.0 + noise)

    def get_sensor_data(self) -> Dict[str, np.ndarray]:
        """Return the current noisy sensor vector for every cell."""
        return {
            "voltage": self._add_noise(self.v_cap),
            "current": self._add_noise(self.i_cell),
            "temperature": self._add_noise(self.temperature),
        }

    def get_true_state(self) -> Dict[str, np.ndarray]:
        """Return the noise-free state (useful for plotting / debugging)."""
        return {
            "voltage": self.v_cap.copy(),
            "current": self.i_cell.copy(),
            "temperature": self.temperature.copy(),
        }

    def ground_truth_fault(self) -> np.ndarray:
        """Boolean array: which cells are *truly* in a faulted state right now."""
        over_temp = self.temperature > TEMP_FAULT_THRESHOLD_C
        over_volt = (
            np.abs(self.v_cap - self.params.v_nominal)
            > VOLT_DEVIATION_FRACTION * self.params.v_nominal
        )
        return over_temp | over_volt


def generate_dataset(
    num_samples: int = 4000,
    seed: int = 1,
    fault_fraction: float = 0.35,
) -> "tuple[np.ndarray, np.ndarray]":
    """Generate a balanced labelled sensor dataset for training the detector.

    Healthy points are sampled around the nominal operating envelope; faulty
    points are sampled from the over-temperature / over-voltage regions so the
    classifier learns a meaningful decision boundary. Returns ``(X, y)`` where
    ``X`` has shape ``(num_samples, 3)`` ([voltage, temperature, current]) and
    ``y`` is 0 (healthy) / 1 (fault).
    """
    rng = np.random.default_rng(seed)
    p = MMCParameters()
    n_fault = int(num_samples * fault_fraction)
    n_ok = num_samples - n_fault

    # The arm current swings over a wide AC range and is *not* by itself a
    # fault indicator, so both classes share the same broad current
    # distribution -- this teaches the classifier to key on temperature and
    # voltage (the real fault signatures) rather than overfit to current.
    i_lo, i_hi = -p.i_ac_amplitude, p.i_ac_amplitude

    # Healthy cluster: nominal voltage, comfortably sub-threshold temperature.
    # Voltage is clipped strictly inside the +-12 % over-voltage band so a
    # healthy (y=0) label never lands in the region ground_truth_fault() would
    # flag as a fault, keeping the training labels self-consistent at any N.
    v_margin = (VOLT_DEVIATION_FRACTION - 0.01) * p.v_nominal
    ok_v = np.clip(
        rng.normal(p.v_nominal, 0.03 * p.v_nominal, n_ok),
        p.v_nominal - v_margin, p.v_nominal + v_margin,
    )
    ok_t = rng.uniform(28.0, TEMP_FAULT_THRESHOLD_C - 8.0, n_ok)
    ok_i = rng.uniform(i_lo, i_hi, n_ok)

    # Faulty cluster: over-temperature and/or large voltage deviation.
    f_hot = rng.uniform(TEMP_FAULT_THRESHOLD_C + 2.0, 130.0, n_fault)
    f_v = p.v_nominal + rng.choice([-1, 1], n_fault) * rng.uniform(
        (VOLT_DEVIATION_FRACTION + 0.01) * p.v_nominal, 0.25 * p.v_nominal, n_fault
    )
    # Half the faults are thermal-only (nominal voltage); half are voltage
    # deviations at a still-rising-but-sub-limit temperature.
    mask = rng.random(n_fault) < 0.5
    f_v = np.where(mask, rng.normal(p.v_nominal, 0.03 * p.v_nominal, n_fault), f_v)
    f_t = np.where(mask, f_hot, rng.uniform(40.0, TEMP_FAULT_THRESHOLD_C - 2.0, n_fault))
    f_i = rng.uniform(i_lo, i_hi, n_fault)

    X = np.vstack(
        [
            np.column_stack([ok_v, ok_t, ok_i]),
            np.column_stack([f_v, f_t, f_i]),
        ]
    ).astype(np.float32)
    y = np.concatenate([np.zeros(n_ok), np.ones(n_fault)]).astype(np.int64)

    # Shuffle so the two clusters are interleaved.
    order = rng.permutation(num_samples)
    return X[order], y[order]


def export_dataset_csv(path: "str | Path", num_samples: int = 4000, seed: int = 1) -> Path:
    """Write a labelled dataset to ``path`` as CSV (voltage,temperature,current,fault)."""
    import csv

    X, y = generate_dataset(num_samples=num_samples, seed=seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["voltage_V", "temperature_C", "current_A", "fault"])
        for (v, t, i), label in zip(X, y):
            writer.writerow([f"{v:.4f}", f"{t:.4f}", f"{i:.4f}", int(label)])
    return path


if __name__ == "__main__":  # pragma: no cover - smoke test
    sim = MMCSimulator(num_cells=10)
    for _ in range(600):
        sim.step()
    print("Healthy/faulted cells after 600 steps:", sim.ground_truth_fault())
    print("Temperatures:", np.round(sim.temperature, 1))
