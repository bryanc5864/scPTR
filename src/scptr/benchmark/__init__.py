"""Benchmark module for scPTR — validation and evaluation metrics."""

from ._halflife_correlation import correlate_with_halflives
from ._enrichment import are_enrichment, nmd_enrichment
from ._robustness import subsampling_robustness
from ._consistency import cross_dataset_consistency

__all__ = [
    "correlate_with_halflives",
    "are_enrichment",
    "nmd_enrichment",
    "subsampling_robustness",
    "cross_dataset_consistency",
]
