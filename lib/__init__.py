"""
Core library for probes-for-diffuse-control.

This package provides the core functionality for:
- Loading and formatting MMLU data
- Generating completions via VLLM API
- Extracting neural activations
- Training and evaluating probes
- Visualization
"""

from . import data
from . import generation
from . import activations
from . import probes
from . import visualization

__all__ = [
    "data",
    "generation",
    "activations",
    "probes",
    "visualization",
]

