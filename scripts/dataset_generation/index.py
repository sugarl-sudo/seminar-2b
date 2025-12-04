#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from datasets import save_index_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate self-referential index datasets.")
    parser.add_argument("--sequence-lengths", type=int, nargs="+", required=True, help="List of n values.")
    parser.add_argument("--m", type=int, required=True, help="Number of previous outputs to sum.")
    parser.add_argument("--train-samples", type=int, default=100000, help="Training sample count.")
    parser.add_argument("--test-samples", type=int, default=1000, help="Test sample count.")
    parser.add_argument("--train-seed", type=int, default=42, help="Seed for training split.")
    parser.add_argument("--test-seed", type=int, default=43, help="Seed for test split.")
    parser.add_argument("--output-root", type=str, default="data/data/small/index", help="Root output directory.")
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

    for n in sequence_lengths:
        seq_dir = output_root / f"n={n}_m={args.m}"
        seq_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Index] Generating n={n}, m={args.m} into {seq_dir} ...")

        save_index_dataset(
            seq_dir / "data.train",
            num_samples=args.train_samples,
            n=n,
            m=args.m,
            seed=args.train_seed,
            inverse=False,
        )
        save_index_dataset(
            seq_dir / "data.test",
            num_samples=args.test_samples,
            n=n,
            m=args.m,
            seed=args.test_seed,
            inverse=False,
        )

        if not args.no_inverse:
            save_index_dataset(
                seq_dir / "data-inv.train",
                num_samples=args.train_samples,
                n=n,
                m=args.m,
                seed=args.train_seed,
                inverse=True,
            )
            save_index_dataset(
                seq_dir / "data-inv.test",
                num_samples=args.test_samples,
                n=n,
                m=args.m,
                seed=args.test_seed,
                inverse=True,
            )

    print("Index dataset generation complete.")


if __name__ == "__main__":
    main()

