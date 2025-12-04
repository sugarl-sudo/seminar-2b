"""Utility functions for generating arithmetic datasets."""

from .relu import (
    generate_relu_samples,
    save_relu_dataset,
)
from .index import (
    generate_index_samples,
    save_index_dataset,
)
from .square import (
    generate_square_mod_samples,
    save_square_mod_dataset,
)

__all__ = [
    "generate_relu_samples",
    "save_relu_dataset",
    "generate_index_samples",
    "save_index_dataset",
    "generate_square_mod_samples",
    "save_square_mod_dataset",
]

