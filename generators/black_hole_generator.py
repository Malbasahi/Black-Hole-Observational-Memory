"""
black_hole_generator.py
-----------------------
Generates a single clean synthetic black hole image from physical parameters.

This version strengthens visible coupling between physical history variables
and the final rendered image. It is still physics-inspired, not a GR renderer.

Pipeline:
    blank canvas
    -> inclination-aware coordinate system
    -> shadow
    -> photon ring
    -> accretion disk
    -> Doppler asymmetry
    -> turbulence structures
    -> jet / instability / spin visual memory features
    -> clean blurred image
"""

from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict, Any

import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

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

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def random(rng: np.random.Generator = None) -> "BlackHoleParams":
        if rng is None:
            rng = np.random.default_rng()

        return BlackHoleParams(
            mass=float(rng.uniform(0.5, 2.0)),
            spin=float(rng.uniform(0.0, 1.0)),
            disk_intensity=float(rng.uniform(0.3, 1.0)),
            observer_angle=float(rng.uniform(0.0, 75.0)),
            ring_thickness=int(rng.integers(1, 9)),
            noise_level=float(rng.uniform(0.0, 0.20)),
            blur_strength=float(rng.uniform(0.0, 3.0)),
            turbulence=float(rng.uniform(0.0, 1.0)),
        )


# ---------------------------------------------------------------------------
# Coordinate grids
# ---------------------------------------------------------------------------

def _make_grids(size: int) -> Tuple[np.ndarray, np.ndarray]:
    axis = (np.arange(size, dtype=np.float32) - size / 2.0) / (size / 2.0)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    return x, y


def _inclined_radius_and_angle(
    x: np.ndarray,
    y: np.ndarray,
    observer_angle: float,
) -> Tuple[np.ndarray, np.ndarray]:
    angle_rad = np.deg2rad(observer_angle)
    cos_i = max(0.30, float(np.cos(angle_rad)))

    y_disk = y / cos_i

    r_inc = np.sqrt(x**2 + y_disk**2)
    phi_inc = np.arctan2(y_disk, x)

    return r_inc, phi_inc


def _circular_radius_and_angle(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return r, phi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _history_value(
    history_features: Optional[Dict[str, Any]],
    key: str,
    default: float,
) -> float:
    """
    Safely read a scalar history feature.

    This expects optional values such as:
        final_accretion_rate
        recent_accretion_rate
        recent_disk_luminosity
        recent_turbulence_level
        recent_instability_index
        recent_jet_power
        final_spin_evolution

    If not provided, the renderer falls back to BlackHoleParams.
    """
    if history_features is None:
        return float(default)

    if key not in history_features:
        return float(default)

    value = history_features[key]

    if isinstance(value, (list, tuple, np.ndarray)):
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if len(value) == 0:
            return float(default)
        value = float(value[-1])

    return float(np.clip(value, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Component renderers
# ---------------------------------------------------------------------------

def _smooth_shadow_mask(r: np.ndarray, mass: float) -> np.ndarray:
    radius = 0.12 * mass
    edge_width = 0.012

    mask = np.clip((r - radius) / edge_width, 0.0, 1.0)
    return mask.astype(np.float32)


def _photon_ring(
    r: np.ndarray,
    phi: np.ndarray,
    mass: float,
    spin: float,
    observer_angle: float,
    ring_thickness: int,
    size: int,
    instability_memory: float = 0.0,
    jet_memory: float = 0.0,
) -> np.ndarray:
    ring_radius = 0.145 * mass
    sigma = max((ring_thickness / size) * 0.75, 1e-4)

    ring = np.exp(-0.5 * ((r - ring_radius) / sigma) ** 2)

    angle_rad = np.deg2rad(observer_angle)
    asym_strength = 0.35 * spin * np.sin(angle_rad)
    boost_axis = spin * np.pi

    asymmetry = 1.0 + asym_strength * np.cos(phi - boost_axis)

    flare = 1.0 + 0.35 * instability_memory * np.cos(3.0 * phi + 2.0 * spin)
    jet_lensing = 1.0 + 0.25 * jet_memory * np.cos(2.0 * phi - np.pi / 3.0)

    ring = ring * asymmetry * flare * jet_lensing

    return np.clip(ring, 0.0, 1.0).astype(np.float32)


def _turbulence_map(
    r: np.ndarray,
    phi: np.ndarray,
    turbulence: float,
    rng: np.random.Generator,
    instability_memory: float = 0.0,
) -> np.ndarray:
    n_modes = 6 + int(4 * instability_memory)
    turb = np.zeros_like(r, dtype=np.float32)

    phases = rng.uniform(0.0, 2.0 * np.pi, n_modes)
    angular_freqs = rng.integers(3, 16, n_modes)
    radial_freqs = rng.uniform(2.0, 10.0, n_modes)

    for phase, a_freq, r_freq in zip(phases, angular_freqs, radial_freqs):
        turb += np.sin(a_freq * phi + phase) * np.cos(r_freq * np.pi * r)

    sigma = max(0.6, 1.8 - 0.9 * instability_memory + 1.2 * turbulence)
    turb = gaussian_filter(turb, sigma=sigma)

    t_min = float(turb.min())
    t_max = float(turb.max())

    if t_max - t_min < 1e-8:
        return np.zeros_like(turb, dtype=np.float32)

    turb = (turb - t_min) / (t_max - t_min)
    return turb.astype(np.float32)


def _accretion_disk(
    r: np.ndarray,
    phi: np.ndarray,
    mass: float,
    disk_intensity: float,
    observer_angle: float,
    spin: float,
    turbulence: float,
    rng: np.random.Generator,
    accretion_memory: float = 0.0,
    luminosity_memory: float = 0.0,
    turbulence_memory: float = 0.0,
    instability_memory: float = 0.0,
) -> np.ndarray:
    """
    Generate an accretion disk with stronger history coupling.

    Couplings:
        accretion_memory  -> inner disk brightness
        luminosity_memory -> global disk brightness and halo
        turbulence_memory -> irregular disk texture
        instability_memory -> sharp spiral/patch features
    """
    r_inner = 0.15 * mass
    r_outer = 0.62 * mass

    disk_region = (r > r_inner) & (r < r_outer)

    norm_r = np.clip((r - r_inner) / (r_outer - r_inner + 1e-8), 0.0, 1.0)

    inner_boost = 1.0 + 0.90 * accretion_memory * np.exp(-5.0 * norm_r)
    luminosity_boost = 0.75 + 0.85 * luminosity_memory

    radial = disk_intensity * luminosity_boost * inner_boost * np.exp(-3.2 * norm_r)
    radial *= disk_region.astype(np.float32)

    angle_rad = np.deg2rad(observer_angle)

    boost_axis = spin * np.pi
    doppler = 0.5 + 0.5 * np.cos(phi - boost_axis)

    asym_strength = 0.9 * spin * np.sin(angle_rad)
    azimuthal = 1.0 + asym_strength * (doppler - 0.5)

    disk = radial * azimuthal

    effective_turbulence = np.clip(
        0.40 * turbulence + 0.60 * turbulence_memory,
        0.0,
        1.0,
    )

    if effective_turbulence > 0.0:
        turb_map = _turbulence_map(
            r,
            phi,
            effective_turbulence,
            rng,
            instability_memory=instability_memory,
        )

        disk *= 1.0 + 0.85 * effective_turbulence * (turb_map - 0.5)

    if instability_memory > 0.0:
        spiral = np.sin(
            5.0 * phi
            + 10.0 * r
            + 2.0 * np.pi * instability_memory
        )

        spiral = 0.5 + 0.5 * spiral
        spiral *= disk_region.astype(np.float32)

        disk += 0.22 * instability_memory * spiral * np.exp(-1.8 * norm_r)

    return np.clip(disk, 0.0, 1.0).astype(np.float32)


def _background_glow(
    r: np.ndarray,
    disk_intensity: float,
    mass: float,
    luminosity_memory: float = 0.0,
) -> np.ndarray:
    glow_scale = 0.35 * mass
    glow = np.exp(-((r / glow_scale) ** 2))
    glow *= (0.10 + 0.18 * luminosity_memory) * disk_intensity
    return np.clip(glow, 0.0, 1.0).astype(np.float32)


def _jet_feature(
    x: np.ndarray,
    y: np.ndarray,
    spin_memory: float,
    jet_memory: float,
    observer_angle: float,
) -> np.ndarray:
    """
    Add faint bipolar jet-like emission.

    Couplings:
        jet_memory  -> jet brightness and length
        spin_memory -> jet orientation
    """
    if jet_memory <= 0.0:
        return np.zeros_like(x, dtype=np.float32)

    theta = spin_memory * np.pi + np.deg2rad(observer_angle) * 0.25

    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)

    width = 0.035 + 0.025 * (1.0 - jet_memory)
    length = 0.55 + 0.35 * jet_memory

    jet_axis = np.exp(-0.5 * (xr / width) ** 2)
    jet_extent = np.exp(-0.5 * ((np.abs(yr) - 0.35) / length) ** 2)

    central_gap = 1.0 - np.exp(-0.5 * (yr / 0.16) ** 2)

    jet = jet_axis * jet_extent * central_gap
    jet *= 0.28 * jet_memory

    return np.clip(jet, 0.0, 1.0).astype(np.float32)


def _spin_warp_feature(
    r: np.ndarray,
    phi: np.ndarray,
    spin_memory: float,
) -> np.ndarray:
    """
    Adds a subtle spiral-like warp correlated with spin history.
    """
    if spin_memory <= 0.0:
        return np.zeros_like(r, dtype=np.float32)

    warp = np.sin(2.0 * phi + 8.0 * r + 2.0 * np.pi * spin_memory)
    warp = 0.5 + 0.5 * warp

    radial_gate = np.exp(-((r - 0.38) / 0.20) ** 2)

    feature = 0.12 * spin_memory * warp * radial_gate
    return np.clip(feature, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_clean_image(
    params: BlackHoleParams,
    size: int = 128,
    rng: np.random.Generator = None,
    history_features: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Generate one clean synthetic black hole image.

    Optional history_features can inject stronger memory coupling.

    Expected optional keys:
        final_accretion_rate
        recent_accretion_rate
        recent_disk_luminosity
        recent_turbulence_level
        recent_instability_index
        recent_jet_power
        final_spin_evolution
    """
    if rng is None:
        rng = np.random.default_rng()

    x, y = _make_grids(size)

    r_circ, phi_circ = _circular_radius_and_angle(x, y)
    r_inc, phi_inc = _inclined_radius_and_angle(x, y, params.observer_angle)

    accretion_memory = _history_value(
        history_features,
        "recent_accretion_rate",
        params.disk_intensity,
    )

    luminosity_memory = _history_value(
        history_features,
        "recent_disk_luminosity",
        params.disk_intensity,
    )

    turbulence_memory = _history_value(
        history_features,
        "recent_turbulence_level",
        params.turbulence,
    )

    instability_memory = _history_value(
        history_features,
        "recent_instability_index",
        params.turbulence,
    )

    jet_memory = _history_value(
        history_features,
        "recent_jet_power",
        0.0,
    )

    spin_memory = _history_value(
        history_features,
        "final_spin_evolution",
        params.spin,
    )

    shadow_mask = _smooth_shadow_mask(r_circ, params.mass)

    ring = _photon_ring(
        r=r_inc,
        phi=phi_inc,
        mass=params.mass,
        spin=params.spin,
        observer_angle=params.observer_angle,
        ring_thickness=params.ring_thickness,
        size=size,
        instability_memory=instability_memory,
        jet_memory=jet_memory,
    )

    disk = _accretion_disk(
        r=r_inc,
        phi=phi_inc,
        mass=params.mass,
        disk_intensity=params.disk_intensity,
        observer_angle=params.observer_angle,
        spin=params.spin,
        turbulence=params.turbulence,
        rng=rng,
        accretion_memory=accretion_memory,
        luminosity_memory=luminosity_memory,
        turbulence_memory=turbulence_memory,
        instability_memory=instability_memory,
    )

    glow = _background_glow(
        r=r_inc,
        disk_intensity=params.disk_intensity,
        mass=params.mass,
        luminosity_memory=luminosity_memory,
    )

    jet = _jet_feature(
        x=x,
        y=y,
        spin_memory=spin_memory,
        jet_memory=jet_memory,
        observer_angle=params.observer_angle,
    )

    spin_warp = _spin_warp_feature(
        r=r_inc,
        phi=phi_inc,
        spin_memory=spin_memory,
    )

    image = (
        0.72 * disk
        + 0.92 * ring
        + glow
        + jet
        + spin_warp
    )

    image *= shadow_mask

    if params.blur_strength > 0.0:
        image = gaussian_filter(image, sigma=params.blur_strength)

    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    return image


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from PIL import Image

    rng = np.random.default_rng(42)
    params = BlackHoleParams.random(rng)

    history_features = {
        "recent_accretion_rate": 0.85,
        "recent_disk_luminosity": 0.80,
        "recent_turbulence_level": 0.70,
        "recent_instability_index": 0.75,
        "recent_jet_power": 0.65,
        "final_spin_evolution": 0.90,
    }

    img = generate_clean_image(
        params,
        size=128,
        rng=rng,
        history_features=history_features,
    )

    img_uint8 = (img * 255).astype(np.uint8)
    Image.fromarray(img_uint8, mode="L").save("black_hole_test.png")

    print("Generated black_hole_test.png")
    print(params)