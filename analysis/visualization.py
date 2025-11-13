"""
Visualization utilities for model analysis.

This module provides plotting functions for eigenvalue distributions,
dimensionality reduction results, and other analyses.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def setup_plot_style():
    """Set up consistent plotting style."""
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9


def plot_eigenvalues_complex_plane(
    eigenvalues_list: List[np.ndarray],
    labels: Optional[List[str]] = None,
    colors: Optional[List] = None,
    title: str = "Eigenvalue Distributions",
    output_path: Optional[str] = None,
    show_unit_circle: bool = True,
    alpha: float = 0.5,
):
    """
    Plot eigenvalues in the complex plane.

    Args:
        eigenvalues_list: List of eigenvalue arrays
        labels: Labels for each set
        colors: Colors for each set
        title: Plot title
        output_path: Path to save figure
        show_unit_circle: Draw unit circle
        alpha: Transparency
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(eigenvalues_list)))

    for i, evals in enumerate(eigenvalues_list):
        label = labels[i] if labels else f"Model {i + 1}"
        ax.scatter(
            evals.real, evals.imag, alpha=alpha, s=30, color=colors[i], label=label
        )

    if show_unit_circle:
        circle = plt.Circle(
            (0, 0),
            1,
            fill=False,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.5,
        )
        ax.add_patch(circle)

    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if labels:
        ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_embedding_2d(
    X: np.ndarray,
    labels: Optional[np.ndarray] = None,
    color_values: Optional[np.ndarray] = None,
    color_label: str = "Value",
    title: str = "2D Embedding",
    xlabel: str = "Dimension 1",
    ylabel: str = "Dimension 2",
    annotate_indices: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (12, 9),
    cmap: str = "viridis",
):
    """
    Plot 2D embedding with optional coloring and annotations.

    Args:
        X: 2D coordinates (n_samples, 2)
        labels: Text labels for points
        color_values: Values for color mapping
        color_label: Label for colorbar
        title: Plot title
        xlabel, ylabel: Axis labels
        annotate_indices: Indices of points to annotate
        output_path: Path to save figure
        figsize: Figure size
        cmap: Colormap
    """
    fig, ax = plt.subplots(figsize=figsize)

    if color_values is not None:
        valid = ~np.isnan(color_values)
        scatter = ax.scatter(
            X[valid, 0],
            X[valid, 1],
            c=color_values[valid],
            cmap=cmap,
            s=80,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )

        if np.sum(~valid) > 0:
            ax.scatter(
                X[~valid, 0],
                X[~valid, 1],
                color="gray",
                s=80,
                alpha=0.3,
                label="No data",
            )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_label)
    else:
        ax.scatter(
            X[:, 0], X[:, 1], s=80, alpha=0.7, edgecolors="black", linewidths=0.5
        )

    # Add annotations
    if annotate_indices is not None:
        for idx in annotate_indices:
            if idx < len(X):
                label_text = labels[idx] if labels is not None else str(idx)
                ax.annotate(
                    label_text,
                    (X[idx, 0], X[idx, 1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="yellow",
                        alpha=0.7,
                        edgecolor="black",
                    ),
                    arrowprops=dict(
                        arrowstyle="->", connectionstyle="arc3,rad=0", lw=0.8
                    ),
                )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_pca_variance(
    explained_variance: np.ndarray,
    n_components: int = 10,
    output_path: Optional[str] = None,
):
    """
    Plot explained variance by PCA components.

    Args:
        explained_variance: Array of explained variance ratios
        n_components: Number of components to show
        output_path: Path to save figure
    """
    n_components = min(n_components, len(explained_variance))
    indices = np.arange(1, n_components + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Individual variance
    ax1.bar(indices, explained_variance[:n_components], alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("Variance Explained by Each Component")
    ax1.set_xticks(indices)
    ax1.grid(True, alpha=0.3, axis="y")

    # Cumulative variance
    cumulative = np.cumsum(explained_variance[:n_components])
    ax2.plot(indices, cumulative, marker="o", linewidth=2, markersize=8)
    ax2.axhline(0.9, color="red", linestyle="--", label="90% threshold")
    ax2.axhline(0.95, color="orange", linestyle="--", label="95% threshold")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title("Cumulative Variance Explained")
    ax2.set_xticks(indices)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_distance_matrix(
    distance_matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Distance Matrix",
    output_path: Optional[str] = None,
    cmap: str = "viridis",
    figsize: tuple = (10, 8),
):
    """
    Plot distance matrix as heatmap.

    Args:
        distance_matrix: Symmetric distance matrix
        labels: Labels for rows/columns
        title: Plot title
        output_path: Path to save figure
        cmap: Colormap
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(distance_matrix, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, label="Distance")

    if labels:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

    ax.set_title(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_histogram_comparison(
    histograms: List[np.ndarray],
    bins: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Histogram Comparison",
    xlabel: str = "Value",
    ylabel: str = "Count",
    output_path: Optional[str] = None,
    alpha: float = 0.5,
):
    """
    Plot multiple histograms for comparison.

    Args:
        histograms: List of histogram arrays
        bins: Bin edges
        labels: Labels for histograms
        title: Plot title
        xlabel, ylabel: Axis labels
        output_path: Path to save figure
        alpha: Transparency
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    bin_centers = (bins[:-1] + bins[1:]) / 2

    for i, hist in enumerate(histograms):
        label = labels[i] if labels else f"Model {i + 1}"
        ax.plot(bin_centers, hist, alpha=alpha + 0.3, linewidth=2, label=label)
        ax.fill_between(bin_centers, hist, alpha=alpha)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_scatter_matrix(
    data: Dict[str, np.ndarray],
    output_path: Optional[str] = None,
    figsize: tuple = (12, 12),
    alpha: float = 0.5,
):
    """
    Create scatter plot matrix for multiple metrics.

    Args:
        data: Dictionary of {metric_name: values}
        output_path: Path to save figure
        figsize: Figure size
        alpha: Transparency
    """
    metrics = list(data.keys())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(n_metrics, n_metrics, figsize=figsize)

    for i, metric_i in enumerate(metrics):
        for j, metric_j in enumerate(metrics):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histograms
                ax.hist(data[metric_i], bins=20, alpha=0.7, edgecolor="black")
                ax.set_ylabel("Count")
            else:
                # Off-diagonal: scatter plots
                ax.scatter(data[metric_j], data[metric_i], alpha=alpha, s=30)

            if i == n_metrics - 1:
                ax.set_xlabel(metric_j)
            if j == 0:
                ax.set_ylabel(metric_i)

            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def save_figure(fig, output_path: str, dpi: int = 300):
    """
    Save figure with consistent settings.

    Args:
        fig: Matplotlib figure
        output_path: Output file path
        dpi: Resolution
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {output_path}")
