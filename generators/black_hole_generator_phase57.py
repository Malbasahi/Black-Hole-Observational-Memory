"""
black_hole_generator_phase57.py

Phase 5.7 decoupled synthetic black hole generator.

Purpose:
    Redesign weak-target couplings after Phase 5.6 showed that
    spin_evolution and jet_power were recoverable, while accretion_rate,
    disk_luminosity, turbulence_level, and instability_index remained weak.

This is physics-inspired synthetic rendering, not a full GR renderer.
"""

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


@dataclass
class BlackHoleParams:
    mass: float
    spin: float
    disk_intensity: float
    observer_angle: float
    ring_thickness: int
    noise_level: float
    blur_strength: float
    turbulence: float

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def random(rng=None):
        if rng is None:
            rng = np.random.default_rng()
        return BlackHoleParams(
            mass=float(rng.uniform(0.5, 2.0)),
            spin=float(rng.uniform(0.0, 1.0)),
            disk_intensity=float(rng.uniform(0.3, 1.0)),
            observer_angle=float(rng.uniform(0.0, 75.0)),
            ring_thickness=int(rng.integers(1, 9)),
            noise_level=float(rng.uniform(0.0, 0.30)),
            blur_strength=float(rng.uniform(0.0, 3.0)),
            turbulence=float(rng.uniform(0.0, 1.0)),
        )


def summarize_history_for_image(history_df):
    recent = history_df.tail(15)
    final = history_df.iloc[-1]
    return {
        "acc_recent": float(recent["accretion_rate"].mean()),
        "lum_recent": float(recent["disk_luminosity"].mean()),
        "turb_recent": float(recent["turbulence_level"].mean()),
        "inst_recent": float(recent["instability_index"].mean()),
        "jet_recent": float(recent["jet_power"].mean()),
        "spin_final": float(final["spin_evolution"]),
        "spin_recent": float(recent["spin_evolution"].mean()),
    }
