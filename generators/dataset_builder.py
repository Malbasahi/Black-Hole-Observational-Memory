"""
dataset_builder.py
------------------
Phase 1.5 dataset builder.

This version makes the image depend on the generated temporal history.

Old order:
    sample params -> generate image -> generate history

New order:
    sample base params
    -> generate temporal history
    -> derive visible image parameters from recent/final history
    -> generate clean image
    -> corrupt image
    -> save image + metadata + history

This is essential for Phase 3:
    current image -> inferred past/current accretion behavior
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generators.black_hole_generator import BlackHoleParams, generate_clean_image
from generators.corruption import corrupt_image
from generators.history_generator import generate_history


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

def _setup_dirs(output_dir: Path, save_history: bool) -> Dict[str, Path]:
    dirs = {
        "clean": output_dir / "clean",
        "noisy": output_dir / "noisy",
        "metadata": output_dir / "metadata",
    }

    if save_history:
        dirs["history"] = output_dir / "history"

    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    return dirs


def _clear_existing_dataset(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _float32_to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def _save_grayscale_png(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_float32_to_uint8(image), mode="L").save(path)


# ---------------------------------------------------------------------------
# History-conditioned parameter coupling
# ---------------------------------------------------------------------------

def _clip(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi))


def _derive_image_params_from_history(
    base_params: BlackHoleParams,
    history: pd.DataFrame,
    rng: np.random.Generator,
) -> Tuple[BlackHoleParams, Dict[str, float]]:
    """
    Convert temporal history into visible image parameters.

    The key design principle:
        recent/final history must leave visible traces in the current image.

    Couplings:
        final accretion_rate       -> disk_intensity
        recent disk_luminosity     -> disk_intensity boost
        recent turbulence_level    -> visible turbulence
        final spin_evolution       -> spin asymmetry
        recent jet_power           -> spin/asymmetry proxy
        recent instability_index   -> extra turbulence boost
    """

    final_accretion = float(history["accretion_rate"].iloc[-1])
    recent_accretion = float(history["accretion_rate"].tail(10).mean())

    final_luminosity = float(history["disk_luminosity"].iloc[-1])
    recent_luminosity = float(history["disk_luminosity"].tail(10).mean())

    final_turbulence = float(history["turbulence_level"].iloc[-1])
    recent_turbulence = float(history["turbulence_level"].tail(10).mean())

    final_jet_power = float(history["jet_power"].iloc[-1])
    recent_jet_power = float(history["jet_power"].tail(10).mean())

    final_spin = float(history["spin_evolution"].iloc[-1])

    if "instability_index" in history.columns:
        recent_instability = float(history["instability_index"].tail(10).mean())
    else:
        recent_instability = recent_turbulence

    # -----------------------------------------------------------------------
    # Visible image parameters derived from history
    # -----------------------------------------------------------------------

    # Disk brightness should strongly encode current/recent accretion.
    disk_intensity = final_accretion
    disk_intensity = _clip(disk_intensity, 0.30, 1.00)

    # Visible turbulence should encode recent turbulence and instability.
    visible_turbulence = (
        0.55 * recent_turbulence
        + 0.25 * final_turbulence
        + 0.20 * recent_instability
    )
    visible_turbulence += rng.normal(0.0, 0.025)
    visible_turbulence = _clip(visible_turbulence, 0.0, 1.0)

    # Spin/asymmetry should be dominated by spin evolution, with jet contribution.
    visible_spin = (
        0.75 * final_spin
        + 0.25 * recent_jet_power
    )
    visible_spin += rng.normal(0.0, 0.015)
    visible_spin = _clip(visible_spin, 0.0, 1.0)

    # Optional: make high jet power slightly sharpen the visible ring.
    ring_thickness = int(base_params.ring_thickness)
    if recent_jet_power > 0.65 and ring_thickness > 1:
        ring_thickness -= 1
    elif recent_instability > 0.75 and ring_thickness < 8:
        ring_thickness += 1

    derived_params = BlackHoleParams(
        mass=base_params.mass,
        spin=visible_spin,
        disk_intensity=disk_intensity,
        observer_angle=base_params.observer_angle,
        ring_thickness=ring_thickness,
        noise_level=base_params.noise_level,
        blur_strength=base_params.blur_strength,
        turbulence=visible_turbulence,
    )

    coupling_info = {
        "final_accretion_rate": final_accretion,
        "recent_accretion_rate_mean10": recent_accretion,
        "final_disk_luminosity": final_luminosity,
        "recent_disk_luminosity_mean10": recent_luminosity,
        "final_turbulence_level": final_turbulence,
        "recent_turbulence_level_mean10": recent_turbulence,
        "final_jet_power": final_jet_power,
        "recent_jet_power_mean10": recent_jet_power,
        "final_spin_evolution": final_spin,
        "recent_instability_index_mean10": recent_instability,
        "derived_disk_intensity": disk_intensity,
        "derived_visible_turbulence": visible_turbulence,
        "derived_visible_spin": visible_spin,
        "derived_ring_thickness": ring_thickness,
    }

    return derived_params, coupling_info


# ---------------------------------------------------------------------------
# Single sample generation
# ---------------------------------------------------------------------------

def generate_one_sample(
    index: int,
    dirs: Dict[str, Path],
    image_size: int,
    rng: np.random.Generator,
    save_history: bool = True,
    history_T: int = 100,
) -> BlackHoleParams:
    """
    Generate and save one complete history-conditioned sample.
    """
    stem = f"bh_{index:04d}"

    # 1. Sample base parameters.
    base_params = BlackHoleParams.random(rng)

    # 2. Generate history first.
    # These initial values act as the latent physical state from which the
    # visible image parameters will be derived.
    history = generate_history(
        mass=base_params.mass,
        spin=base_params.spin,
        disk_intensity=base_params.disk_intensity,
        turbulence=base_params.turbulence,
        T=history_T,
        rng=rng,
    )

    # 3. Derive visible image parameters from history.
    params, coupling_info = _derive_image_params_from_history(
        base_params=base_params,
        history=history,
        rng=rng,
    )

    # 4. Generate image using history-conditioned parameters.
    clean = generate_clean_image(
        params=params,
        size=image_size,
        rng=rng,
    )

    noisy = corrupt_image(
        clean=clean,
        noise_level=params.noise_level,
        blur_strength=params.blur_strength,
        rng=rng,
        use_sparse=True,
        use_fourier=True,
    )

    _save_grayscale_png(clean, dirs["clean"] / f"{stem}.png")
    _save_grayscale_png(noisy, dirs["noisy"] / f"{stem}_noisy.png")

    # 5. Save history if requested.
    if save_history:
        history.to_csv(dirs["history"] / f"{stem}.csv", index=False)

    # 6. Save metadata.
    metadata = params.to_dict()
    metadata.update(
        {
            "sample_index": index,
            "sample_id": stem,
            "image_size": image_size,
            "has_history": save_history,
            "history_conditioned": True,
            "coupling_version": "1.5",
            "base_params": base_params.to_dict(),
            "history_coupling": coupling_info,
        }
    )

    with open(dirs["metadata"] / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return params


# ---------------------------------------------------------------------------
# Batch builder
# ---------------------------------------------------------------------------

def build_dataset(
    n: int = 1000,
    image_size: int = 128,
    seed: int = 42,
    output_dir: str = "./data/dataset",
    save_history: bool = True,
    history_T: int = 100,
    verbose: bool = True,
    overwrite: bool = True,
) -> None:
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    if image_size <= 0:
        raise ValueError("image_size must be a positive integer.")

    if history_T <= 1:
        raise ValueError("history_T must be greater than 1.")

    output_path = Path(output_dir)

    if overwrite:
        _clear_existing_dataset(output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    dirs = _setup_dirs(output_path, save_history=save_history)

    master_rng = np.random.default_rng(seed)
    child_seeds = master_rng.integers(0, 2**32 - 1, size=n, dtype=np.uint32)

    params_log = []

    iterator = range(1, n + 1)

    if verbose:
        iterator = tqdm(
            iterator,
            desc="Generating history-conditioned black holes",
            unit="sample",
            ncols=90,
        )

    for index in iterator:
        sample_rng = np.random.default_rng(int(child_seeds[index - 1]))

        params = generate_one_sample(
            index=index,
            dirs=dirs,
            image_size=image_size,
            rng=sample_rng,
            save_history=save_history,
            history_T=history_T,
        )

        row = params.to_dict()
        row["sample_index"] = index
        row["sample_id"] = f"bh_{index:04d}"
        row["history_conditioned"] = True
        params_log.append(row)

    params_table = pd.DataFrame(params_log)
    params_table.to_csv(output_path / "params_table.csv", index=False)

    manifest = {
        "dataset_name": "phase1_5_history_conditioned_black_hole_dataset",
        "version": "1.5",
        "n_samples": n,
        "image_size": image_size,
        "seed": seed,
        "save_history": save_history,
        "history_T": history_T if save_history else None,
        "history_conditioned": True,
        "coupling_description": {
            "disk_intensity": "derived from final/recent accretion_rate and recent disk_luminosity",
            "turbulence": "derived from recent turbulence_level and instability_index",
            "spin": "derived from final spin_evolution and recent jet_power",
            "ring_thickness": "slightly adjusted using recent jet_power and instability_index",
        },
        "image_format": "grayscale_png",
        "pixel_range": [0.0, 1.0],
        "filename_pattern": {
            "clean": "clean/bh_XXXX.png",
            "noisy": "noisy/bh_XXXX_noisy.png",
            "metadata": "metadata/bh_XXXX.json",
            "history": "history/bh_XXXX.csv" if save_history else None,
        },
        "parameter_ranges": {
            "mass": [0.5, 2.0],
            "spin": [0.0, 1.0],
            "disk_intensity": [0.3, 1.0],
            "observer_angle": [0.0, 75.0],
            "ring_thickness": [1, 8],
            "noise_level": [0.0, 0.30],
            "blur_strength": [0.0, 3.0],
            "turbulence": [0.0, 1.0],
        },
        "corruption_model": {
            "uv_sparse_sampling": True,
            "fourier_phase_amplitude_distortion": True,
            "gaussian_thermal_noise": True,
            "atmospheric_blur": True,
        },
    }

    with open(output_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if verbose:
        print(f"\nDataset complete: {n} samples -> {output_path}")
        print(f"  clean/        {n} PNG files")
        print(f"  noisy/        {n} PNG files")
        print(f"  metadata/     {n} JSON files")
        if save_history:
            print(f"  history/      {n} CSV files ({history_T} timesteps each)")
        print("  manifest.json")
        print("  params_table.csv")
        print("  history-conditioned coupling: ON")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Phase 1.5 history-conditioned black hole dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="./data/dataset")
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--history-T", type=int, default=100)
    parser.add_argument("--no-overwrite", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    build_dataset(
        n=args.n,
        image_size=args.size,
        seed=args.seed,
        output_dir=args.out,
        save_history=not args.no_history,
        history_T=args.history_T,
        verbose=True,
        overwrite=not args.no_overwrite,
    )