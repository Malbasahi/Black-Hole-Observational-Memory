"""
history_generator.py
--------------------
Generates synthetic temporal histories for each black hole sample.

The generated history is designed to be correlated with the final static
image parameters. This makes the later "memory reconstruction" model more
meaningful.

Output columns:
    time_step
    accretion_rate
    disk_luminosity
    turbulence_level
    jet_power
    spin_evolution
"""

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _clip01(x: np.ndarray) -> np.ndarray:
    """Clip values to [0, 1]."""
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _smooth_series(x: np.ndarray, passes: int = 2) -> np.ndarray:
    """
    Lightweight smoothing using repeated nearest-neighbor averaging.

    Avoids abrupt jumps while keeping the script dependency-light.
    """
    x = x.astype(np.float32).copy()

    for _ in range(passes):
        x[1:-1] = 0.25 * x[:-2] + 0.50 * x[1:-1] + 0.25 * x[2:]

    return x


def _ou_bridge(
    T: int,
    start: float,
    end: float,
    mu: float,
    sigma: float,
    theta: float,
    rng: np.random.Generator,
    lo: float = 0.0,
    hi: float = 1.0,
) -> np.ndarray:
    """
    Mean-reverting stochastic series softly constrained to end near a target.

    This is better than a free OU process because the final state should agree
    with the image-generating parameters.
    """
    if T <= 1:
        raise ValueError("T must be greater than 1.")

    x = np.empty(T, dtype=np.float32)
    x[0] = float(np.clip(start, lo, hi))

    for t in range(1, T):
        noise = rng.normal(0.0, sigma)
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) + noise
        x[t] = float(np.clip(x[t], lo, hi))

    # Bridge correction: smoothly steer the trajectory toward the desired end.
    correction = np.linspace(0.0, end - x[-1], T, dtype=np.float32)
    x = x + correction

    x = _smooth_series(x, passes=2)
    x[-1] = end

    return np.clip(x, lo, hi).astype(np.float32)


# ---------------------------------------------------------------------------
# Individual physical histories
# ---------------------------------------------------------------------------

def _gen_accretion_rate(
    T: int,
    disk_intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Accretion history.

    The final value is strongly coupled to disk_intensity, so a bright current
    disk corresponds to a high recent accretion state.
    """
    final = float(np.clip(disk_intensity + rng.normal(0.0, 0.015), 0.05, 1.0))
    start = float(rng.uniform(0.08, 0.95))
    mu = float(0.65 * disk_intensity + 0.35 * start)

    series = _ou_bridge(
        T=T,
        start=start,
        end=final,
        mu=mu,
        sigma=float(rng.uniform(0.015, 0.06)),
        theta=float(rng.uniform(0.04, 0.13)),
        rng=rng,
        lo=0.05,
        hi=1.0,
    )

    return series


def _gen_disk_luminosity(
    accretion: np.ndarray,
    mass: float,
    disk_intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Disk luminosity proxy.

    Larger accretion increases luminosity. Larger mass softens normalized
    brightness slightly. Final value remains coupled to disk_intensity.
    """
    mass_factor = 1.0 / np.sqrt(max(mass, 1e-6))
    raw = accretion * mass_factor

    raw = raw / (raw.max() + 1e-8)
    raw = 0.75 * raw + 0.25 * disk_intensity

    jitter = rng.normal(0.0, 0.015, len(raw)).astype(np.float32)
    luminosity = _smooth_series(raw + jitter, passes=1)

    return _clip01(luminosity)


def _gen_turbulence_level(
    T: int,
    turbulence: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Turbulence history.

    Final value is coupled to the image turbulence parameter.
    """
    final = float(np.clip(turbulence + rng.normal(0.0, 0.02), 0.0, 1.0))
    start = float(rng.uniform(0.0, 1.0))
    mu = float(0.75 * turbulence + 0.25 * start)

    series = _ou_bridge(
        T=T,
        start=start,
        end=final,
        mu=mu,
        sigma=float(rng.uniform(0.02, 0.08)),
        theta=float(rng.uniform(0.05, 0.18)),
        rng=rng,
        lo=0.0,
        hi=1.0,
    )

    return series


def _gen_jet_power(
    spin: float,
    accretion: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Jet power proxy.

    Uses a simplified Blandford-Znajek-like relationship:

        jet_power ∝ spin^2 * accretion_rate

    This is not a full physical jet model, but it creates a meaningful
    nonlinear relationship between spin, accretion, and jet output.
    """
    spin_factor = float(np.clip(spin, 0.0, 1.0)) ** 2
    raw = spin_factor * accretion

    if raw.max() > 1e-8:
        raw = raw / max(1.0, raw.max())

    jitter = rng.normal(0.0, 0.01, len(raw)).astype(np.float32)
    jet = _smooth_series(raw + jitter, passes=1)

    return _clip01(jet)


def _gen_spin_evolution(
    T: int,
    spin: float,
    accretion: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Slowly varying spin history.

    Spin should not fluctuate wildly. The final value is coupled to the
    current spin parameter.
    """
    final = float(np.clip(spin + rng.normal(0.0, 0.01), 0.0, 1.0))
    start = float(np.clip(spin + rng.normal(0.0, 0.12), 0.0, 1.0))

    x = np.empty(T, dtype=np.float32)
    x[0] = start

    for t in range(1, T):
        torque = 0.004 * accretion[t] * (final - x[t - 1])
        drift = 0.01 * (final - x[t - 1])
        noise = rng.normal(0.0, 0.002)
        x[t] = float(np.clip(x[t - 1] + torque + drift + noise, 0.0, 1.0))

    correction = np.linspace(0.0, final - x[-1], T, dtype=np.float32)
    x = x + correction
    x = _smooth_series(x, passes=2)
    x[-1] = final

    return _clip01(x)


def _gen_instability_index(
    accretion: np.ndarray,
    turbulence: np.ndarray,
    jet_power: np.ndarray,
) -> np.ndarray:
    """
    Extra derived feature.

    Represents how dynamically unstable the disk environment is.
    Useful later as an auxiliary target or diagnostic.
    """
    acc_grad = np.abs(np.gradient(accretion))
    turb_grad = np.abs(np.gradient(turbulence))

    raw = (
        0.45 * turbulence
        + 0.25 * jet_power
        + 0.20 * acc_grad / (acc_grad.max() + 1e-8)
        + 0.10 * turb_grad / (turb_grad.max() + 1e-8)
    )

    return _clip01(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_history(
    mass: float,
    spin: float,
    disk_intensity: float,
    turbulence: float,
    T: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Generate one synthetic temporal history.

    Parameters
    ----------
    mass:
        Black hole mass multiplier.
    spin:
        Current spin-like parameter.
    disk_intensity:
        Current disk brightness parameter.
    turbulence:
        Current turbulence parameter.
    T:
        Number of time steps.
    rng:
        Optional NumPy random generator.

    Returns
    -------
    DataFrame with temporal physical proxy variables.
    """
    if T <= 1:
        raise ValueError("T must be greater than 1.")

    if rng is None:
        rng = np.random.default_rng()

    mass = float(np.clip(mass, 0.5, 2.0))
    spin = float(np.clip(spin, 0.0, 1.0))
    disk_intensity = float(np.clip(disk_intensity, 0.3, 1.0))
    turbulence = float(np.clip(turbulence, 0.0, 1.0))

    accretion = _gen_accretion_rate(
        T=T,
        disk_intensity=disk_intensity,
        rng=rng,
    )

    luminosity = _gen_disk_luminosity(
        accretion=accretion,
        mass=mass,
        disk_intensity=disk_intensity,
        rng=rng,
    )

    turb = _gen_turbulence_level(
        T=T,
        turbulence=turbulence,
        rng=rng,
    )

    jet = _gen_jet_power(
        spin=spin,
        accretion=accretion,
        rng=rng,
    )

    spin_ev = _gen_spin_evolution(
        T=T,
        spin=spin,
        accretion=accretion,
        rng=rng,
    )

    instability = _gen_instability_index(
        accretion=accretion,
        turbulence=turb,
        jet_power=jet,
    )

    return pd.DataFrame(
        {
            "time_step": np.arange(T, dtype=int),
            "accretion_rate": np.round(accretion, 4),
            "disk_luminosity": np.round(luminosity, 4),
            "turbulence_level": np.round(turb, 4),
            "jet_power": np.round(jet, 4),
            "spin_evolution": np.round(spin_ev, 4),
            "instability_index": np.round(instability, 4),
        }
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    df = generate_history(
        mass=1.2,
        spin=0.7,
        disk_intensity=0.8,
        turbulence=0.6,
        T=100,
        rng=rng,
    )

    print(df.head())
    print(df.tail())

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    from pathlib import Path
    import matplotlib.pyplot as plt

    OUTPUT_DIR = Path("./history_test_outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    df = generate_history(
        mass=1.2,
        spin=0.7,
        disk_intensity=0.8,
        turbulence=0.6,
        T=100,
        rng=rng,
    )

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------

    csv_path = OUTPUT_DIR / "history_test.csv"
    df.to_csv(csv_path, index=False)

    # -----------------------------------------------------------------------
    # Save plot
    # -----------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 6))

    cols = [
        "accretion_rate",
        "disk_luminosity",
        "turbulence_level",
        "jet_power",
        "spin_evolution",
        "instability_index",
    ]

    for col in cols:
        ax.plot(df["time_step"], df[col], label=col, linewidth=1.8)

    ax.set_title("Synthetic Black Hole Temporal History")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Normalized Value")
    ax.set_ylim(-0.05, 1.05)

    ax.grid(alpha=0.25)
    ax.legend()

    plot_path = OUTPUT_DIR / "history_test_plot.png"

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)

    plt.close(fig)

    # -----------------------------------------------------------------------
    # Console output
    # -----------------------------------------------------------------------

    print("\nHistory generation test complete.")
    print(f"CSV saved to:  {csv_path}")
    print(f"Plot saved to: {plot_path}")