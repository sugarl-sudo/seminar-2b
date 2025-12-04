from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .common import Sample, write_pairs


def square_mod_forward(inputs: list[int], modulus: int = 19) -> list[int]:
    """Compute forward task used in the square-mod dataset."""
    outputs: list[int] = []
    for idx, value in enumerate(inputs):
        if idx == 0:
            outputs.append(value)
        else:
            prev = outputs[idx - 1]
            y_val = (value**2 + prev**2) % modulus - (modulus // 2)
            outputs.append(y_val)
    return outputs


def generate_square_mod_samples(
    num_samples: int,
    sequence_length: int,
    seed: int | None = None,
    *,
    value_min: int = -9,
    value_max: int = 9,
    modulus: int = 19,
    inverse: bool = False,
) -> Iterable[Sample]:
    """Yield samples for the square-mod dataset."""
    if value_min > value_max:
        raise ValueError("value_min must be <= value_max.")
    if modulus <= 0:
        raise ValueError("modulus must be positive.")

    rng = np.random.default_rng(seed)
    high = value_max + 1

    for _ in range(num_samples):
        inputs = rng.integers(value_min, high, size=sequence_length).tolist()
        outputs = square_mod_forward(inputs, modulus=modulus)
        if inverse:
            outputs = outputs[::-1]
        yield inputs, outputs


def save_square_mod_dataset(
    output_path: str | Path,
    num_samples: int,
    sequence_length: int,
    seed: int | None = None,
    **kwargs,
) -> None:
    """Persist the square-mod dataset to disk."""
    path = Path(output_path)
    samples = generate_square_mod_samples(
        num_samples=num_samples,
        sequence_length=sequence_length,
        seed=seed,
        **kwargs,
    )
    write_pairs(samples, path)

