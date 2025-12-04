from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

from .common import Sample, write_pairs


def generate_self_ref_index_sample(n: int, m: int, rng: random.Random) -> Sample:
    """Generate a single self-referential index sample."""
    if n < 2:
        raise ValueError("n must be at least 2.")
    if m <= 0:
        raise ValueError("m must be positive.")

    x = [rng.randint(1, n - 1) for _ in range(n)]
    y: list[int] = []

    for idx in range(n):
        if idx == 0:
            pointer = x[0] % n
        else:
            start = max(0, idx - m)
            current_sum = sum(y[start:idx])
            pointer = current_sum % n
        y_val = x[pointer]
        y.append(y_val)

    return x, y


def generate_index_samples(
    num_samples: int,
    n: int,
    m: int,
    seed: int | None = None,
    *,
    inverse: bool = False,
) -> Iterable[Sample]:
    """Yield samples for the self-referential index dataset."""
    rng = random.Random(seed)
    for _ in range(num_samples):
        inputs, outputs = generate_self_ref_index_sample(n, m, rng)
        if inverse:
            outputs = outputs[::-1]
        yield inputs, outputs


def save_index_dataset(
    output_path: str | Path,
    num_samples: int,
    n: int,
    m: int,
    seed: int | None = None,
    *,
    inverse: bool = False,
) -> None:
    """Write index dataset to disk."""
    path = Path(output_path)
    samples = generate_index_samples(
        num_samples=num_samples,
        n=n,
        m=m,
        seed=seed,
        inverse=inverse,
    )
    write_pairs(samples, path)

