from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .common import Sample, validate_permutation, write_pairs


def relu_forward(inputs: Sequence[int]) -> list[int]:
    """Compute cumulative ReLU outputs for ``inputs``."""
    outputs = [0] * len(inputs)
    if not inputs:
        return outputs

    outputs[0] = max(0, inputs[0])
    for idx in range(1, len(inputs)):
        outputs[idx] = max(0, outputs[idx - 1] + inputs[idx])
    return outputs


def generate_relu_samples(
    num_samples: int,
    sequence_length: int,
    seed: int | None = None,
    *,
    inverse: bool = False,
    permutation: Sequence[int] | None = None,
    value_min: int = -9,
    value_max: int = 9,
) -> Iterable[Sample]:
    """
    Yield ``num_samples`` pairs following the Chain-of-Thought ReLU task.
    """
    if value_min > value_max:
        raise ValueError("value_min must be <= value_max")

    if permutation is not None:
        validate_permutation(permutation, sequence_length)
        if inverse:
            raise ValueError("inverse cannot be combined with permutation.")

    rng = np.random.default_rng(seed)
    high = value_max + 1

    for _ in range(num_samples):
        inputs = rng.integers(value_min, high, size=sequence_length).tolist()
        outputs = relu_forward(inputs)

        if permutation is not None:
            outputs = [outputs[idx] for idx in permutation]
        elif inverse:
            outputs = outputs[::-1]

        yield inputs, outputs


def save_relu_dataset(
    output_path: str | Path,
    num_samples: int,
    sequence_length: int,
    seed: int | None = None,
    **kwargs,
) -> None:
    """Materialize ReLU dataset to ``output_path``."""
    path = Path(output_path)
    samples = generate_relu_samples(
        num_samples=num_samples,
        sequence_length=sequence_length,
        seed=seed,
        **kwargs,
    )
    write_pairs(samples, path)

