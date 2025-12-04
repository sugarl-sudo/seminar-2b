from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple


Sample = Tuple[Sequence[int], Sequence[int]]


def ensure_directory(path: Path) -> None:
    """Ensure the parent directory of ``path`` exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def format_pair(inputs: Sequence[int], outputs: Sequence[int]) -> str:
    """Format a single input/output pair as ``\"x : y\"``."""
    input_str = " ".join(map(str, inputs))
    output_str = " ".join(map(str, outputs))
    return f"{input_str} : {output_str}"


def write_pairs(samples: Iterable[Sample], output_path: Path) -> None:
    """Write iterable of samples to ``output_path``."""
    ensure_directory(output_path)
    with output_path.open("w", encoding="utf-8") as file:
        for inputs, outputs in samples:
            file.write(format_pair(inputs, outputs) + "\n")


def validate_permutation(permutation: Sequence[int], expected_length: int) -> None:
    """Validate permutation matches ``expected_length``."""
    if len(permutation) != expected_length:
        raise ValueError(
            f"Permutation length ({len(permutation)}) "
            f"must match sequence length ({expected_length})."
        )
    if set(permutation) != set(range(expected_length)):
        raise ValueError(
            "Permutation must contain unique integers from 0 to sequence_length - 1."
        )

