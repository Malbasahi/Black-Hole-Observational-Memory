"""
run_generation.py
-----------------
Top-level entry point for Phase 1 dataset generation.

Runs:
    1. Dataset generation
    2. Optional validation
    3. Visual report generation through validate_dataset.py

Quick start:
    python run_generation.py
    python run_generation.py --n 100 --quick
    python run_generation.py --n 10000 --size 256 --seed 7
"""

import argparse
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 Synthetic Black Hole Dataset Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Image resolution in pixels",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./data/dataset",
        help="Root dataset output directory",
    )
    parser.add_argument(
        "--outputs",
        type=str,
        default="./outputs",
        help="Directory for validation report images",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Skip temporal history CSV generation",
    )
    parser.add_argument(
        "--history-T",
        type=int,
        default=100,
        help="Time steps per history CSV",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip post-generation validation",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shortcut: 50 samples, no history, skip validation",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not delete an existing output dataset before generation",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_args(args: argparse.Namespace) -> None:
    if args.n <= 0:
        raise ValueError("--n must be greater than 0.")

    if args.size <= 0:
        raise ValueError("--size must be greater than 0.")

    if args.history_T <= 1:
        raise ValueError("--history-T must be greater than 1.")


def _print_banner(args: argparse.Namespace) -> None:
    history_status = "OFF" if args.no_history else f"ON ({args.history_T} timesteps)"
    validate_status = "OFF" if args.skip_validate else "ON"
    overwrite_status = "OFF" if args.no_overwrite else "ON"

    print("\n" + "=" * 68)
    print("Phase 1 Synthetic Black Hole Dataset Generator")
    print("=" * 68)
    print(f"samples        : {args.n}")
    print(f"resolution     : {args.size} x {args.size} px")
    print(f"seed           : {args.seed}")
    print(f"output dir     : {args.out}")
    print(f"reports dir    : {args.outputs}")
    print(f"history        : {history_status}")
    print(f"validation     : {validate_status}")
    print(f"overwrite      : {overwrite_status}")
    print("=" * 68 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root))

    args = parse_args()

    if args.quick:
        args.n = 50
        args.no_history = True
        args.skip_validate = True
        print("[quick mode] n=50, no history, validation skipped")

    _validate_args(args)
    _print_banner(args)

    from generators.dataset_builder import build_dataset

    t0 = time.time()

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

    elapsed = time.time() - t0
    ms_per_sample = (elapsed / args.n) * 1000.0

    print("\nGeneration summary")
    print("-" * 68)
    print(f"total time      : {elapsed:.2f} seconds")
    print(f"time per sample : {ms_per_sample:.2f} ms")
    print("-" * 68)

    if not args.skip_validate:
        from validate_dataset import validate_dataset

        validate_dataset(
            dataset_dir=args.out,
            n_check=min(args.n, 200),
            outputs_dir=args.outputs,
        )

    print("\nPhase 1 complete.\n")


if __name__ == "__main__":
    main()