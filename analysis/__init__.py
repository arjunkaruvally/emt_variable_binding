"""
Analysis Package for EMT Variable Binding Experiments.

This package provides modular tools for analyzing trained RNN models,
including eigenvalue analysis, dimensionality reduction, and topological
data analysis.

Modules:
    eigenvalue_analysis: Core eigenvalue distribution analysis
    distance_metrics: Distance metrics for comparing models
    dimensionality_reduction: PCA, t-SNE, MDS, etc.
    visualization: Plotting utilities
    persistent_homology: Topological data analysis
"""

__version__ = "1.0.0"

from . import (
    dimensionality_reduction,
    distance_metrics,
    eigenvalue_analysis,
    visualization,
)

__all__ = [
    "eigenvalue_analysis",
    "distance_metrics",
    "dimensionality_reduction",
    "visualization",
]
