"""
validate_dataset.py
-------------------
Phase 1 validation script.

Checks:
    1. Completeness
    2. Parameter diversity
    3. Image quality
    4. History sanity
    5. Visual reports

Usage:
    python validate_dataset.py --dataset ./data/dataset --n-check 200 --outputs ./outputs
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import kstest


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_manifest(dataset_dir: Path) -> dict:
    manifest_path = dataset_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_params_table(dataset_dir: Path) -> pd.DataFrame:
    params_path = dataset_dir / "params_table.csv"

    if not params_path.exists():
        raise FileNotFoundError(f"Missing params table: {params_path}")

    return pd.read_csv(params_path)


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def _check_completeness(
    dataset_dir: Path,
    n: int,
    save_history: bool,
) -> bool:
    """Verify expected files exist for samples 1..n."""
    expected = {
        "clean": ("clean", "bh_{i:04d}.png"),
        "noisy": ("noisy", "bh_{i:04d}_noisy.png"),
        "metadata": ("metadata", "bh_{i:04d}.json"),
    }

    if save_history:
        expected["history"] = ("history", "bh_{i:04d}.csv")

    missing = []

    for _, (folder, pattern) in expected.items():
        for i in range(1, n + 1):
            path = dataset_dir / folder / pattern.format(i=i)
            if not path.exists():
                missing.append(str(path))

    if missing:
        print(f"  [WARN] {len(missing)} missing files. First few:")
        for item in missing[:8]:
            print(f"         {item}")
        return False

    print(f"  [OK]  All expected files present for {n} samples.")
    return True


def _check_diversity(params: pd.DataFrame) -> None:
    """KS test against expected uniform parameter ranges."""
    ranges = {
        "mass": (0.5, 2.0),
        "spin": (0.0, 1.0),
        "disk_intensity": (0.3, 1.0),
        "observer_angle": (0.0, 75.0),
        "ring_thickness": (1.0, 8.0),
        "noise_level": (0.0, 0.30),
        "blur_strength": (0.0, 3.0),
        "turbulence": (0.0, 1.0),
    }

    print("\n  Parameter diversity (KS test vs uniform):")

    for col, (lo, hi) in ranges.items():
        if col not in params.columns:
            print(f"    {col:20s}  [SKIP] missing column")
            continue

        values = params[col].dropna().to_numpy(dtype=float)

        if values.size == 0:
            print(f"    {col:20s}  [WARN] empty column")
            continue

        normed = (values - lo) / (hi - lo)
        normed = np.clip(normed, 0.0, 1.0)

        stat, p_value = kstest(normed, "uniform")
        flag = "OK" if p_value > 0.01 else "LOW p-value"

        print(f"    {col:20s}  KS={stat:.4f}  p={p_value:.4f}  {flag}")


def _check_image_quality(
    dataset_dir: Path,
    n_check: int = 100,
    seed: int = 123,
) -> None:
    """Compare brightness, contrast, and clean/noisy difference."""
    clean_dir = dataset_dir / "clean"
    noisy_dir = dataset_dir / "noisy"

    clean_files = sorted(clean_dir.glob("*.png"))
    total = len(clean_files)

    if total == 0:
        print("\n  [WARN] No clean images found.")
        return

    rng = np.random.default_rng(seed)
    n_eval = min(n_check, total)
    indices = rng.choice(np.arange(1, total + 1), size=n_eval, replace=False)

    clean_means = []
    noisy_means = []
    clean_stds = []
    noisy_stds = []
    mae_vals = []
    snr_vals = []

    for i in indices:
        clean_path = clean_dir / f"bh_{i:04d}.png"
        noisy_path = noisy_dir / f"bh_{i:04d}_noisy.png"

        if not clean_path.exists() or not noisy_path.exists():
            continue

        clean = np.asarray(Image.open(clean_path), dtype=np.float32) / 255.0
        noisy = np.asarray(Image.open(noisy_path), dtype=np.float32) / 255.0

        clean_means.append(float(clean.mean()))
        noisy_means.append(float(noisy.mean()))
        clean_stds.append(float(clean.std()))
        noisy_stds.append(float(noisy.std()))

        diff = np.abs(clean - noisy)
        mae = float(diff.mean())
        mae_vals.append(mae)

        signal_power = float(np.mean(clean**2)) + 1e-8
        noise_power = float(np.mean((clean - noisy) ** 2)) + 1e-8
        snr_vals.append(10.0 * np.log10(signal_power / noise_power))

    if not clean_means:
        print("\n  [WARN] No valid clean/noisy pairs found.")
        return

    print(f"\n  Image quality ({len(clean_means)} samples):")
    print(f"    Clean mean      = {np.mean(clean_means):.4f}")
    print(f"    Clean std       = {np.mean(clean_stds):.4f}")
    print(f"    Noisy mean      = {np.mean(noisy_means):.4f}")
    print(f"    Noisy std       = {np.mean(noisy_stds):.4f}")
    print(f"    Mean abs diff   = {np.mean(mae_vals):.4f}")
    print(
        f"    Mean SNR        = {np.mean(snr_vals):.1f} dB "
        f"(range {np.min(snr_vals):.1f} to {np.max(snr_vals):.1f} dB)"
    )


def _check_history_sanity(
    dataset_dir: Path,
    n_check: int = 50,
    save_history: bool = True,
) -> None:
    """Verify history CSV columns, bounds, and NaNs."""
    if not save_history:
        print("\n  [SKIP] History disabled in manifest.")
        return

    hist_dir = dataset_dir / "history"
    csvs = sorted(hist_dir.glob("*.csv"))

    if not csvs:
        print("\n  [WARN] Manifest expects history, but no history files were found.")
        return

    expected_cols = {
        "time_step",
        "accretion_rate",
        "disk_luminosity",
        "turbulence_level",
        "jet_power",
        "spin_evolution",
    }

    optional_cols = {"instability_index"}
    errors = 0
    checked = min(n_check, len(csvs))

    for path in csvs[:checked]:
        df = pd.read_csv(path)

        if not expected_cols.issubset(df.columns):
            errors += 1
            continue

        numeric_cols = [
            col for col in df.columns
            if col != "time_step" and (col in expected_cols or col in optional_cols)
        ]

        numeric = df[numeric_cols]

        has_nan = numeric.isnull().any().any()
        below = (numeric < -0.05).any().any()
        above = (numeric > 1.05).any().any()

        if has_nan or below or above:
            errors += 1

    flag = "OK" if errors == 0 else "WARN"
    print(f"\n  History sanity ({checked} checked): {errors} anomalies found {flag}")


# ---------------------------------------------------------------------------
# Visual reports
# ---------------------------------------------------------------------------

def _make_contact_sheet(
    dataset_dir: Path,
    output_path: Path,
    n_cols: int = 8,
    n_rows: int = 4,
) -> None:
    """Save a contact sheet showing clean/noisy image pairs."""
    clean_dir = dataset_dir / "clean"
    noisy_dir = dataset_dir / "noisy"

    total = len(sorted(clean_dir.glob("*.png")))
    n = min(n_rows * n_cols, total)

    if n == 0:
        print("  [SKIP] No images available for contact sheet.")
        return

    fig = plt.figure(
        figsize=(n_cols * 2.5, n_rows * 4.4),
        facecolor="#0a0a0a",
    )

    grid = gridspec.GridSpec(
        n_rows * 2,
        n_cols,
        figure=fig,
        hspace=0.05,
        wspace=0.05,
    )

    for idx in range(n):
        i = idx + 1
        row_clean = (idx // n_cols) * 2
        row_noisy = row_clean + 1
        col = idx % n_cols

        clean_path = clean_dir / f"bh_{i:04d}.png"
        noisy_path = noisy_dir / f"bh_{i:04d}_noisy.png"

        if not clean_path.exists() or not noisy_path.exists():
            continue

        clean = np.asarray(Image.open(clean_path))
        noisy = np.asarray(Image.open(noisy_path))

        ax_clean = fig.add_subplot(grid[row_clean, col])
        ax_clean.imshow(clean, cmap="inferno", vmin=0, vmax=255)
        ax_clean.axis("off")
        if col == 0:
            ax_clean.set_ylabel("clean", color="white", fontsize=8)

        ax_noisy = fig.add_subplot(grid[row_noisy, col])
        ax_noisy.imshow(noisy, cmap="inferno", vmin=0, vmax=255)
        ax_noisy.axis("off")
        if col == 0:
            ax_noisy.set_ylabel("noisy", color="white", fontsize=8)

    fig.suptitle(
        "Black Hole Dataset - Phase 1 Contact Sheet "
        "(top: clean | bottom: noisy)",
        color="white",
        fontsize=13,
        y=0.995,
    )

    fig.savefig(output_path, bbox_inches="tight", dpi=100, facecolor="#0a0a0a")
    plt.close(fig)

    print(f"\n  Contact sheet saved: {output_path}")


def _make_param_distributions(
    params: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save histogram grid of physical parameters."""
    cols = [
        "mass",
        "spin",
        "disk_intensity",
        "observer_angle",
        "ring_thickness",
        "noise_level",
        "blur_strength",
        "turbulence",
    ]

    cols = [col for col in cols if col in params.columns]

    if not cols:
        print("  [SKIP] No parameter columns found for distribution plot.")
        return

    n_cols = 4
    n_rows = int(np.ceil(len(cols) / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(16, 3.5 * n_rows),
        facecolor="#0a0a0a",
    )

    axes = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes, cols):
        ax.set_facecolor("#111")
        ax.hist(
            params[col].dropna().to_numpy(),
            bins=40,
            color="#e07b39",
            edgecolor="none",
            alpha=0.85,
        )
        ax.set_title(col, color="white", fontsize=10)
        ax.tick_params(colors="grey")

        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    for ax in axes[len(cols):]:
        ax.axis("off")

    fig.suptitle(
        "Parameter Distributions",
        color="white",
        fontsize=13,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=100, facecolor="#0a0a0a")
    plt.close(fig)

    print(f"  Parameter distribution plot saved: {output_path}")


def _make_history_sample(
    dataset_dir: Path,
    output_path: Path,
    n_samples: int = 6,
) -> None:
    """Plot temporal histories for a few sample histories."""
    hist_dir = dataset_dir / "history"
    csvs = sorted(hist_dir.glob("*.csv"))

    if not csvs:
        print("  [SKIP] No history files found; skipping history preview.")
        return

    chosen = csvs[:min(n_samples, len(csvs))]

    fig, axes = plt.subplots(
        len(chosen),
        1,
        figsize=(12, 2.5 * len(chosen)),
        facecolor="#0a0a0a",
        sharex=True,
    )

    axes = np.atleast_1d(axes)

    colors = [
        "#e07b39",
        "#4fc3f7",
        "#a5d6a7",
        "#ce93d8",
        "#ffcc80",
        "#f48fb1",
    ]

    cols = [
        "accretion_rate",
        "disk_luminosity",
        "turbulence_level",
        "jet_power",
        "spin_evolution",
        "instability_index",
    ]

    for ax, csv_path in zip(axes, chosen):
        ax.set_facecolor("#111")

        df = pd.read_csv(csv_path)
        x = df["time_step"].to_numpy()

        for color, col in zip(colors, cols):
            if col in df.columns:
                y = df[col].to_numpy()
                ax.plot(
                    x,
                    y,
                    color=color,
                    lw=1.2,
                    alpha=0.85,
                    label=col,
                )

        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(csv_path.stem, color="grey", fontsize=8)
        ax.tick_params(colors="grey")

        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    axes[0].legend(
        loc="upper right",
        fontsize=7,
        facecolor="#222",
        labelcolor="white",
        framealpha=0.6,
    )

    axes[-1].set_xlabel("time step", color="white")

    fig.suptitle("Temporal Histories - Sample Preview", color="white", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=100, facecolor="#0a0a0a")
    plt.close(fig)

    print(f"  History preview saved: {output_path}")


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

def validate_dataset(
    dataset_dir: str,
    n_check: int = 200,
    outputs_dir: str = "./outputs",
) -> None:
    dataset_path = Path(dataset_dir)
    output_path = Path(outputs_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Validating dataset: {dataset_path}")
    print(f"{'=' * 60}")

    manifest = _load_manifest(dataset_path)

    n_samples = int(manifest["n_samples"])
    image_size = manifest["image_size"]
    seed = manifest["seed"]
    save_history = bool(manifest.get("save_history", True))

    print(
        f"  Manifest: {n_samples} samples, "
        f"size={image_size}px, seed={seed}, "
        f"history={'ON' if save_history else 'OFF'}"
    )

    n_eval = min(n_samples, n_check)

    print("\n[1] Completeness")
    _check_completeness(dataset_path, n_eval, save_history=save_history)

    print("\n[2] Diversity")
    params = _load_params_table(dataset_path)
    _check_diversity(params)

    print("\n[3] Image quality")
    _check_image_quality(dataset_path, n_check=n_eval)

    print("\n[4] History sanity")
    _check_history_sanity(
        dataset_path,
        n_check=min(n_samples, 50),
        save_history=save_history,
    )

    print("\n[5] Generating visual reports ...")
    _make_contact_sheet(dataset_path, output_path / "contact_sheet.png")
    _make_param_distributions(params, output_path / "param_distributions.png")

    if save_history:
        _make_history_sample(dataset_path, output_path / "history_sample.png")

    print(f"\n{'=' * 60}")
    print(f"  Validation complete. Reports saved to: {output_path}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Phase 1 synthetic black hole dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/dataset",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--n-check",
        type=int,
        default=200,
        help="Number of samples to verify",
    )
    parser.add_argument(
        "--outputs",
        type=str,
        default="./outputs",
        help="Directory for report images",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_dataset(
        dataset_dir=args.dataset,
        n_check=args.n_check,
        outputs_dir=args.outputs,
    )