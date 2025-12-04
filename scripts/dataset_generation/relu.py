#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from gen_data import save_relu_dataset


def _parse_permutation(raw: str | None) -> Sequence[int] | None:
    if raw is None or raw.strip() == "":
        return None
    parts = [chunk.strip() for chunk in raw.split(",")]
    try:
        return [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("Permutation must be a comma-separated list of integers.") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate ReLU datasets.")
    parser.add_argument("--sequence-lengths", type=int, nargs="+", required=True, help="Sequence lengths to generate.")
    parser.add_argument("--train-samples", type=int, default=100000, help="Number of training samples.")
    parser.add_argument("--test-samples", type=int, default=1000, help="Number of test samples.")
    parser.add_argument("--train-seed", type=int, default=42, help="Random seed for training split.")
    parser.add_argument("--test-seed", type=int, default=123, help="Random seed for test split.")
    parser.add_argument("--output-root", type=str, default="data/data/small/relu", help="Root directory for outputs.")
    parser.add_argument("--value-min", type=int, default=-9, help="Minimum input integer value (inclusive).")
    parser.add_argument("--value-max", type=int, default=9, help="Maximum input integer value (inclusive).")
    parser.add_argument(
        "--no-inverse",
        action="store_true",
        help="Skip generation of reversed-output datasets (always skipped when permutation is provided).",
    )
    parser.add_argument(
        "--permutation",
        type=str,
        default=None,
        help='Comma separated permutation list (e.g. "1,0,2").',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    sequence_lengths = sorted(set(args.sequence_lengths))
    output_root = Path(args.output_root)

    try:
        permutation = _parse_permutation(args.permutation)
    except ValueError as exc:
        parser.error(str(exc))
        return

    for seq_len in sequence_lengths:
        seq_dir = output_root / f"n={seq_len}"
        seq_dir.mkdir(parents=True, exist_ok=True)

        print(f"[ReLU] Generating n={seq_len} train/test splits under {seq_dir} ...")

        save_relu_dataset(
            seq_dir / "data.train",
            num_samples=args.train_samples,
            sequence_length=seq_len,
            seed=args.train_seed,
            inverse=False,
            permutation=permutation,
            value_min=args.value_min,
            value_max=args.value_max,
        )
        save_relu_dataset(
            seq_dir / "data.test",
            num_samples=args.test_samples,
            sequence_length=seq_len,
            seed=args.test_seed,
            inverse=False,
            permutation=permutation,
            value_min=args.value_min,
            value_max=args.value_max,
        )

        if args.no_inverse:
            continue

        if permutation is not None:
            print("  Skipping inverse outputs because permutation was provided.")
            continue

        save_relu_dataset(
            seq_dir / "data-inv.train",
            num_samples=args.train_samples,
            sequence_length=seq_len,
            seed=args.train_seed,
            inverse=True,
            permutation=None,
            value_min=args.value_min,
            value_max=args.value_max,
        )
        save_relu_dataset(
            seq_dir / "data-inv.test",
            num_samples=args.test_samples,
            sequence_length=seq_len,
            seed=args.test_seed,
            inverse=True,
            permutation=None,
            value_min=args.value_min,
            value_max=args.value_max,
        )

    print("ReLU dataset generation complete.")


if __name__ == "__main__":
    main()

