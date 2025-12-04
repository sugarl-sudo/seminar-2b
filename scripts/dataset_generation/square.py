#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from gen_data import save_square_mod_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate square-mod datasets.")
    parser.add_argument("--sequence-lengths", type=int, nargs="+", required=True, help="Sequence lengths to generate.")
    parser.add_argument("--train-samples", type=int, default=100000, help="Number of training samples.")
    parser.add_argument("--test-samples", type=int, default=1000, help="Number of test samples.")
    parser.add_argument("--train-seed", type=int, default=42, help="Seed for training split.")
    parser.add_argument("--test-seed", type=int, default=43, help="Seed for test split.")
    parser.add_argument("--output-root", type=str, default="data/data/small/square", help="Root directory for outputs.")
    parser.add_argument("--value-min", type=int, default=-9, help="Minimum input integer value (inclusive).")
    parser.add_argument("--value-max", type=int, default=9, help="Maximum input integer value (inclusive).")
    parser.add_argument("--modulus", type=int, default=19, help="Modulus used in the recurrence (default: 19).")
    parser.add_argument(
        "--no-inverse",
        action="store_true",
        help="Skip generation of reversed-output datasets.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    sequence_lengths = sorted(set(args.sequence_lengths))
    output_root = Path(args.output_root)

    for seq_len in sequence_lengths:
        seq_dir = output_root / f"n={seq_len}"
        seq_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Square] Generating n={seq_len} into {seq_dir} ...")

        save_square_mod_dataset(
            seq_dir / "data.train",
            num_samples=args.train_samples,
            sequence_length=seq_len,
            seed=args.train_seed,
            value_min=args.value_min,
            value_max=args.value_max,
            modulus=args.modulus,
            inverse=False,
        )
        save_square_mod_dataset(
            seq_dir / "data.test",
            num_samples=args.test_samples,
            sequence_length=seq_len,
            seed=args.test_seed,
            value_min=args.value_min,
            value_max=args.value_max,
            modulus=args.modulus,
            inverse=False,
        )

        if not args.no_inverse:
            save_square_mod_dataset(
                seq_dir / "data-inv.train",
                num_samples=args.train_samples,
                sequence_length=seq_len,
                seed=args.train_seed,
                value_min=args.value_min,
                value_max=args.value_max,
                modulus=args.modulus,
                inverse=True,
            )
            save_square_mod_dataset(
                seq_dir / "data-inv.test",
                num_samples=args.test_samples,
                sequence_length=seq_len,
                seed=args.test_seed,
                value_min=args.value_min,
                value_max=args.value_max,
                modulus=args.modulus,
                inverse=True,
            )

    print("Square dataset generation complete.")


if __name__ == "__main__":
    main()

