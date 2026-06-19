"""
black_hole_generator_phase5.py
------------------------------
Phase 5 strongly-coupled synthetic black hole generator.

This generator is designed after Phase 4.5 showed that several physical
histories were weakly recoverable. It strengthens generator-side visual
coupling for turbulence_level, instability_index, jet_power, and spin_evolution.

This is physics-inspired, not a full GR renderer.
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
    def random(rng: np.random.Generator = None) -> "BlackHoleParams":
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


def _normalize01(arr):
    arr = np.asarray(arr, dtype=np.float32)
    mn = float(arr.min())
    mx = float(arr.max())

    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)

    return ((arr - mn) / (mx - mn)).astype(np.float32)


def _make_grids(size: int):
    axis = (np.arange(size, dtype=np.float32) - size / 2.0) / (size / 2.0)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    return x, y


def _inclined_radius_and_angle(x, y, observer_angle):
    angle_rad = np.deg2rad(observer_angle)
    cos_i = max(0.30, float(np.cos(angle_rad)))

    y_disk = y / cos_i
    r_inc = np.sqrt(x**2 + y_disk**2)
    phi_inc = np.arctan2(y_disk, x)

    return r_inc, phi_inc


def _circular_radius_and_angle(x, y):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return r, phi


def _smooth_random_curve(rng, T, base, amplitude, smooth_sigma=4.0):
    noise = rng.normal(0.0, 1.0, T).astype(np.float32)
    noise = gaussian_filter(noise, sigma=smooth_sigma)

    if noise.max() > noise.min():
        noise = (noise - noise.min()) / (noise.max() - noise.min())

    noise = 2.0 * noise - 1.0
    curve = base + amplitude * noise

    return np.clip(curve, 0.0, 1.0).astype(np.float32)


def generate_physical_histories(params, rng, T=100):
    t = np.linspace(0.0, 1.0, T, dtype=np.float32)

    acc_base = np.clip(
        0.35 + 0.45 * params.disk_intensity + rng.normal(0, 0.08),
        0.05,
        0.95,
    )
    acc_amp = rng.uniform(0.12, 0.38)

    accretion_rate = _smooth_random_curve(
        rng,
        T,
        acc_base,
        acc_amp,
        smooth_sigma=rng.uniform(3.0, 8.0),
    )

    instability_base = np.clip(
        0.25 + 0.45 * params.turbulence + rng.normal(0, 0.10),
        0.0,
        1.0,
    )

    instability_index = _smooth_random_curve(
        rng,
        T,
        instability_base,
        rng.uniform(0.15, 0.45),
        smooth_sigma=2.0,
    )

    for _ in range(rng.integers(2, 6)):
        center = rng.uniform(0.05, 0.95)
        width = rng.uniform(0.025, 0.085)
        amp = rng.uniform(0.08, 0.30)
        instability_index += amp * np.exp(-0.5 * ((t - center) / width) ** 2)

    instability_index = np.clip(instability_index, 0.0, 1.0).astype(np.float32)

    turbulence_level = (
        0.45
        * _smooth_random_curve(
            rng,
            T,
            params.turbulence,
            rng.uniform(0.12, 0.35),
            smooth_sigma=3.0,
        )
        + 0.45 * instability_index
        + 0.10 * rng.random(T)
    )
    turbulence_level = gaussian_filter(turbulence_level, sigma=1.25)
    turbulence_level = np.clip(turbulence_level, 0.0, 1.0).astype(np.float32)

    disk_luminosity = (
        0.62 * accretion_rate
        + 0.22 * params.disk_intensity
        + 0.16 * instability_index
    )
    disk_luminosity = gaussian_filter(disk_luminosity, sigma=1.5)
    disk_luminosity = np.clip(disk_luminosity, 0.0, 1.0).astype(np.float32)

    jet_power = (
        0.42 * params.spin
        + 0.28 * accretion_rate
        + 0.30 * instability_index
    )
    jet_power = gaussian_filter(jet_power, sigma=2.0)
    jet_power = np.clip(jet_power, 0.0, 1.0).astype(np.float32)

    drift = rng.uniform(-0.16, 0.16) * (t - 0.5)
    spin_evolution = (
        params.spin
        + drift
        + 0.12 * gaussian_filter(jet_power - jet_power.mean(), sigma=6.0)
    )
    spin_evolution = np.clip(spin_evolution, 0.0, 1.0).astype(np.float32)

    return pd.DataFrame(
        {
            "time": np.arange(T, dtype=np.int32),
            "accretion_rate": accretion_rate,
            "disk_luminosity": disk_luminosity,
            "turbulence_level": turbulence_level,
            "instability_index": instability_index,
            "jet_power": jet_power,
            "spin_evolution": spin_evolution,
        }
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


def generate_clean_image_phase5(params, history_df, size=128, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    h = summarize_history_for_image(history_df)

    x, y = _make_grids(size)
    r_circ, _ = _circular_radius_and_angle(x, y)

    spin_shift = 0.030 * (h["spin_final"] - 0.5)
    x_shifted = x - spin_shift

    r_inc, phi_inc = _inclined_radius_and_angle(
        x_shifted,
        y,
        params.observer_angle,
    )

    shadow = np.clip(
        (r_circ - 0.12 * params.mass) / (0.012 + 0.004 * h["jet_recent"]),
        0.0,
        1.0,
    )

    warp = 0.014 * h["spin_recent"] * np.sin(
        phi_inc - h["spin_recent"] * np.pi
    )
    warp += 0.010 * h["inst_recent"] * np.sin(
        3.0 * phi_inc + 2.0 * h["spin_recent"] * np.pi
    )

    ring_radius = 0.145 * params.mass + warp
    sigma = max(
        ((params.ring_thickness + 3.0 * h["jet_recent"]) / size) * 0.72,
        1e-4,
    )

    ring = np.exp(-0.5 * ((r_inc - ring_radius) / sigma) ** 2)

    angle_rad = np.deg2rad(params.observer_angle)
    asymmetry = 1.0 + (
        0.35 * h["spin_recent"] * np.sin(angle_rad)
        + 0.28 * h["jet_recent"]
    ) * np.cos(phi_inc - h["spin_recent"] * np.pi)

    ring *= asymmetry

    r_inner = 0.15 * params.mass
    r_outer = 0.62 * params.mass * (1.0 + 0.05 * h["lum_recent"])

    disk_region = (r_inc > r_inner) & (r_inc < r_outer)
    norm_r = np.clip(
        (r_inc - r_inner) / (r_outer - r_inner + 1e-8),
        0.0,
        1.0,
    )

    disk = (0.18 + 0.82 * h["lum_recent"]) * np.exp(
        -(2.5 + 1.4 * h["acc_recent"]) * norm_r
    )
    disk *= disk_region.astype(np.float32)

    doppler = 0.5 + 0.5 * np.cos(phi_inc - h["spin_recent"] * np.pi)
    disk *= 1.0 + (
        0.35 + 0.80 * h["spin_recent"]
    ) * np.sin(angle_rad) * (doppler - 0.5)

    n_modes = int(5 + 10 * h["turb_recent"] + 6 * h["inst_recent"])
    turb = np.zeros_like(r_inc, dtype=np.float32)

    phases = rng.uniform(0.0, 2.0 * np.pi, n_modes)
    angular_freqs = rng.integers(3, 18, n_modes)
    radial_freqs = rng.uniform(2.0, 11.0, n_modes)

    for phase, a_freq, r_freq in zip(phases, angular_freqs, radial_freqs):
        turb += np.sin(a_freq * phi_inc + phase) * np.cos(
            r_freq * np.pi * r_inc + 0.35 * phase
        )

    turb = gaussian_filter(
        turb,
        sigma=max(0.35, 2.2 - 1.4 * h["turb_recent"] - 0.8 * h["inst_recent"]),
    )
    turb = _normalize01(turb)

    disk *= 1.0 + (0.25 + 1.15 * h["turb_recent"]) * (turb - 0.5)

    clumps = np.zeros_like(r_inc, dtype=np.float32)

    for _ in range(int(2 + 12 * h["inst_recent"])):
        rr = rng.uniform(0.17 * params.mass, 0.56 * params.mass)
        pp = rng.uniform(-np.pi, np.pi)
        width_r = rng.uniform(0.012, 0.045) * (1.0 + h["turb_recent"])
        width_p = rng.uniform(0.08, 0.24)
        amp = rng.uniform(0.15, 0.70) * h["inst_recent"]

        dphi = np.angle(np.exp(1j * (phi_inc - pp)))
        clumps += (
            amp
            * np.exp(-0.5 * ((r_inc - rr) / width_r) ** 2)
            * np.exp(-0.5 * (dphi / width_p) ** 2)
        )

    disk += (
        (0.20 + 1.35 * h["inst_recent"])
        * np.clip(clumps, 0.0, 1.0)
        * max(float(disk.max()), 1e-8)
    )

    tilt = (h["spin_recent"] - 0.5) * 0.32
    x_rot = x * np.cos(tilt) - y * np.sin(tilt)
    y_rot = x * np.sin(tilt) + y * np.cos(tilt)

    width = 0.025 + 0.045 * h["jet_recent"]
    length_scale = 0.42 + 0.25 * h["jet_recent"]

    upper = (
        np.exp(-0.5 * (x_rot / width) ** 2)
        * np.exp(-np.maximum(y_rot, 0.0) / length_scale)
        * (y_rot > 0.05 * params.mass)
    )

    lower = (
        np.exp(-0.5 * (x_rot / (width * 1.15)) ** 2)
        * np.exp(-np.maximum(-y_rot, 0.0) / (length_scale * 0.85))
        * (y_rot < -0.05 * params.mass)
    )

    jet = (
        h["jet_recent"]
        * (0.35 + 0.65 * np.sin(angle_rad))
        * (upper + 0.75 * lower)
    )
    jet = gaussian_filter(jet, sigma=1.0 + 1.5 * h["jet_recent"])

    glow_scale = 0.35 * params.mass * (1.0 + 0.12 * h["jet_recent"])
    glow = np.exp(-((r_inc / glow_scale) ** 2))
    glow *= 0.08 + 0.18 * h["lum_recent"]

    image = (
        0.72 * np.clip(disk, 0.0, 1.0)
        + 0.88 * np.clip(ring, 0.0, 1.0)
        + 0.48 * np.clip(jet, 0.0, 1.0)
        + glow
    )

    image *= shadow

    if params.blur_strength > 0.0:
        image = gaussian_filter(image, sigma=min(params.blur_strength, 2.2))

    return np.clip(image, 0.0, 1.0).astype(np.float32)


def corrupt_image(clean, params, rng):
    image = clean.astype(np.float32).copy()

    Fimg = np.fft.fftshift(np.fft.fft2(image))

    h, w = image.shape
    yy, xx = np.mgrid[-1:1:complex(h), -1:1:complex(w)]
    rr = np.sqrt(xx**2 + yy**2)

    keep_strength = 0.55 + 0.35 * rng.random()
    uv_mask = np.exp(-(rr / keep_strength) ** 2)
    uv_mask *= rng.uniform(0.70, 1.0, size=image.shape)

    phase_noise = rng.normal(
        0.0,
        params.noise_level * 0.45,
        size=image.shape,
    )

    Fimg_corrupt = Fimg * uv_mask * np.exp(1j * phase_noise)
    image = np.real(np.fft.ifft2(np.fft.ifftshift(Fimg_corrupt))).astype(np.float32)

    if params.blur_strength > 0.0:
        image = gaussian_filter(image, sigma=0.55 * params.blur_strength)

    if params.noise_level > 0.0:
        image += rng.normal(
            0.0,
            params.noise_level,
            image.shape,
        ).astype(np.float32)

    return np.clip(image, 0.0, 1.0).astype(np.float32)
