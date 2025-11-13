"""
Persistent Homology Analysis of Eigenvalue Distributions.

This module uses topological data analysis (TDA) to study the structure of
eigenvalue distributions in the complex plane. Persistent homology captures
multi-scale topological features (connected components, loops, voids) that
traditional methods might miss.

Requirements:
    pip install ripser persim scikit-tda

Key concepts:
- Treat each model's eigenvalues as a point cloud in C ≅ R²
- Compute persistence diagrams capturing topological features at different scales
- Compare models using bottleneck/Wasserstein distances between diagrams
- Visualize persistence diagrams and landscapes
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from tqdm import tqdm

try:
    from persim import bottleneck, plot_diagrams, wasserstein
    from ripser import ripser

    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    print(
        "Warning: ripser/persim not available. Install with: pip install ripser persim"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Persistent homology analysis of eigenvalue distributions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--summary_file",
        type=str,
        required=True,
        help="Path to evaluation_summary.json file",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="persistence_analysis",
        help="Output directory for plots and results",
    )

    parser.add_argument(
        "--seq_length",
        type=int,
        default=8,
        help="Sequence length to analyze (default: 8)",
    )

    parser.add_argument(
        "--max_dimension",
        type=int,
        default=1,
        help="Maximum homology dimension to compute (0=components, 1=loops, 2=voids)",
    )

    parser.add_argument(
        "--distance_metric",
        type=str,
        choices=["bottleneck", "wasserstein"],
        default="bottleneck",
        help="Distance metric for comparing persistence diagrams",
    )

    parser.add_argument(
        "--embedding_method",
        type=str,
        choices=["mds", "tsne", "pca"],
        default="mds",
        help="Method for embedding distance matrix",
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize eigenvalues to unit circle before computing persistence",
    )

    parser.add_argument(
        "--color_by",
        type=str,
        choices=["none", "accuracy", "h0_births", "h1_deaths"],
        default="accuracy",
        help="Metric to use for coloring points",
    )

    return parser.parse_args()


def load_eigenvalues_from_summary(summary_file, seq_length=8):
    """
    Load eigenvalues from evaluation summary.

    Returns:
        models: List of dicts with model_name, eigenvalues, and metrics
    """
    with open(summary_file) as f:
        summary = json.load(f)

    seq_key = f"seq{seq_length}"
    eigenvalue_key = f"eigenvalues_rnn_{seq_key}"

    models = []
    for model_data in summary["metrics"]:
        if eigenvalue_key not in model_data:
            continue

        # Eigenvalues stored as [real, imag] pairs
        evals_list = model_data[eigenvalue_key]
        eigenvalues = np.array([complex(e[0], e[1]) for e in evals_list])

        models.append(
            {
                "model_name": model_data["model_name"],
                "eigenvalues": eigenvalues,
                "accuracy": model_data.get("accuracy", np.nan),
                "frob_error": model_data.get(f"frob_relative_{seq_key}", np.nan),
                "spectral_error": model_data.get(
                    f"spectral_error_deg_{seq_key}", np.nan
                ),
            }
        )

    return models


def compute_persistence_diagram(eigenvalues, max_dimension=1, normalize=False):
    """
    Compute persistent homology of eigenvalue point cloud.

    Args:
        eigenvalues: Complex array of eigenvalues
        max_dimension: Maximum homology dimension to compute
        normalize: If True, normalize to unit circle

    Returns:
        diagrams: List of persistence diagrams (one per dimension)
    """
    if not RIPSER_AVAILABLE:
        raise ImportError("ripser not available. Install with: pip install ripser")

    # Convert to 2D points in R^2
    points = np.column_stack([eigenvalues.real, eigenvalues.imag])

    # Normalize if requested
    if normalize:
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / (norms + 1e-10)

    # Compute persistent homology using Vietoris-Rips complex
    result = ripser(points, maxdim=max_dimension)

    return result["dgms"]


def persistence_statistics(diagrams):
    """
    Extract statistical features from persistence diagrams.

    Returns:
        dict with various topological features
    """
    stats = {}

    for dim, dgm in enumerate(diagrams):
        # Remove points at infinity
        dgm_finite = dgm[np.isfinite(dgm).all(axis=1)]

        if len(dgm_finite) == 0:
            stats[f"h{dim}_count"] = 0
            stats[f"h{dim}_total_persistence"] = 0.0
            stats[f"h{dim}_max_persistence"] = 0.0
            stats[f"h{dim}_mean_birth"] = 0.0
            stats[f"h{dim}_mean_death"] = 0.0
            continue

        births = dgm_finite[:, 0]
        deaths = dgm_finite[:, 1]
        persistences = deaths - births

        stats[f"h{dim}_count"] = len(dgm_finite)
        stats[f"h{dim}_total_persistence"] = np.sum(persistences)
        stats[f"h{dim}_max_persistence"] = np.max(persistences)
        stats[f"h{dim}_mean_persistence"] = np.mean(persistences)
        stats[f"h{dim}_mean_birth"] = np.mean(births)
        stats[f"h{dim}_mean_death"] = np.mean(deaths)

        # Betti numbers at different scales
        if len(births) > 0:
            scales = np.linspace(0, np.max(deaths), 10)
            for i, scale in enumerate(scales):
                betti = np.sum((births <= scale) & (deaths > scale))
                stats[f"h{dim}_betti_scale{i}"] = betti

    return stats


def compute_diagram_distance(dgm1, dgm2, metric="bottleneck"):
    """
    Compute distance between two persistence diagrams.

    Args:
        dgm1, dgm2: Persistence diagrams (list of arrays, one per dimension)
        metric: 'bottleneck' or 'wasserstein'

    Returns:
        distance: Scalar distance value
    """
    if not RIPSER_AVAILABLE:
        raise ImportError("persim not available")

    # Compute distance for each dimension and sum
    total_distance = 0.0

    for d1, d2 in zip(dgm1, dgm2):
        # Remove infinite points
        d1_finite = d1[np.isfinite(d1).all(axis=1)]
        d2_finite = d2[np.isfinite(d2).all(axis=1)]

        if len(d1_finite) == 0 and len(d2_finite) == 0:
            continue

        if metric == "bottleneck":
            dist = bottleneck(d1_finite, d2_finite)
        else:  # wasserstein
            dist = wasserstein(d1_finite, d2_finite)

        total_distance += dist

    return total_distance


def create_persistence_distance_matrix(diagrams, metric="bottleneck"):
    """
    Compute pairwise distances between persistence diagrams.

    Args:
        diagrams: List of persistence diagrams
        metric: Distance metric to use

    Returns:
        distance_matrix: Symmetric distance matrix
    """
    n = len(diagrams)
    distance_matrix = np.zeros((n, n))

    print(f"Computing {n}x{n} distance matrix using {metric} distance...")
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            dist = compute_diagram_distance(diagrams[i], diagrams[j], metric)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix


def plot_example_persistence_diagrams(models, diagrams, output_path, n_examples=6):
    """
    Plot example persistence diagrams from different models.
    """
    if not RIPSER_AVAILABLE:
        print("Skipping persistence diagram plots (ripser not available)")
        return

    n_examples = min(n_examples, len(models))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Select diverse examples (evenly spaced)
    indices = np.linspace(0, len(models) - 1, n_examples, dtype=int)

    for idx, ax in zip(indices, axes):
        plot_diagrams(diagrams[idx], ax=ax)
        model_name = models[idx]["model_name"]
        accuracy = models[idx]["accuracy"]
        ax.set_title(f"{model_name}\nAccuracy: {accuracy:.4f}", fontsize=10)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved example diagrams to: {output_path}")
    plt.close()


def plot_persistence_embedding(
    X, models, method="MDS", color_values=None, color_label="Accuracy", output_path=None
):
    """
    Plot 2D embedding of persistence distance matrix.
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    if color_values is not None:
        valid = ~np.isnan(color_values)
        scatter = ax.scatter(
            X[valid, 0],
            X[valid, 1],
            c=color_values[valid],
            cmap="viridis",
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

    ax.set_xlabel(f"{method} Dimension 1")
    ax.set_ylabel(f"{method} Dimension 2")
    ax.set_title(f"Persistent Homology: {method} Embedding of Models")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved embedding plot to: {output_path}")
    plt.close()


def plot_topological_features(models, stats_list, output_path):
    """
    Plot distributions of topological features across models.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Extract feature arrays
    h0_counts = [s["h0_count"] for s in stats_list]
    h0_total_pers = [s["h0_total_persistence"] for s in stats_list]
    h0_max_pers = [s["h0_max_persistence"] for s in stats_list]

    h1_counts = [s.get("h1_count", 0) for s in stats_list]
    h1_total_pers = [s.get("h1_total_persistence", 0) for s in stats_list]
    h1_max_pers = [s.get("h1_max_persistence", 0) for s in stats_list]

    accuracies = [m["accuracy"] for m in models]

    # Plot H0 features
    axes[0, 0].scatter(accuracies, h0_counts, alpha=0.6)
    axes[0, 0].set_xlabel("Accuracy")
    axes[0, 0].set_ylabel("H0 Feature Count")
    axes[0, 0].set_title("Connected Components vs Accuracy")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(accuracies, h0_total_pers, alpha=0.6)
    axes[0, 1].set_xlabel("Accuracy")
    axes[0, 1].set_ylabel("H0 Total Persistence")
    axes[0, 1].set_title("H0 Total Persistence vs Accuracy")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].hist(h0_max_pers, bins=20, alpha=0.7, edgecolor="black")
    axes[0, 2].set_xlabel("H0 Max Persistence")
    axes[0, 2].set_ylabel("Count")
    axes[0, 2].set_title("Distribution of H0 Max Persistence")
    axes[0, 2].grid(True, alpha=0.3)

    # Plot H1 features
    axes[1, 0].scatter(accuracies, h1_counts, alpha=0.6)
    axes[1, 0].set_xlabel("Accuracy")
    axes[1, 0].set_ylabel("H1 Feature Count")
    axes[1, 0].set_title("Loops vs Accuracy")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(accuracies, h1_total_pers, alpha=0.6)
    axes[1, 1].set_xlabel("Accuracy")
    axes[1, 1].set_ylabel("H1 Total Persistence")
    axes[1, 1].set_title("H1 Total Persistence vs Accuracy")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].hist(h1_max_pers, bins=20, alpha=0.7, edgecolor="black")
    axes[1, 2].set_xlabel("H1 Max Persistence")
    axes[1, 2].set_ylabel("Count")
    axes[1, 2].set_title("Distribution of H1 Max Persistence")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved topological features plot to: {output_path}")
    plt.close()


def save_statistics_summary(models, stats_list, output_path):
    """
    Save summary statistics to JSON file.
    """
    summary = {"n_models": len(models), "statistics": []}

    for model, stats in zip(models, stats_list):
        entry = {
            "model_name": model["model_name"],
            "accuracy": float(model["accuracy"]),
            **{k: float(v) for k, v in stats.items()},
        }
        summary["statistics"].append(entry)

    # Add aggregate statistics
    all_stats = {}
    for key in stats_list[0].keys():
        values = [s[key] for s in stats_list]
        all_stats[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    summary["aggregate_statistics"] = all_stats

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved statistics summary to: {output_path}")


def main():
    args = parse_args()

    if not RIPSER_AVAILABLE:
        print("\nERROR: This script requires ripser and persim.")
        print("Install with: pip install ripser persim scikit-tda")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 70}")
    print("PERSISTENT HOMOLOGY ANALYSIS OF EIGENVALUE DISTRIBUTIONS")
    print(f"{'=' * 70}\n")

    # Load eigenvalues
    print(f"Loading eigenvalues from: {args.summary_file}")
    models = load_eigenvalues_from_summary(args.summary_file, args.seq_length)
    print(f"Loaded {len(models)} models")

    # Compute persistence diagrams
    print(f"\nComputing persistence diagrams (max dimension = {args.max_dimension})...")
    diagrams = []
    stats_list = []

    for model in tqdm(models):
        dgm = compute_persistence_diagram(
            model["eigenvalues"],
            max_dimension=args.max_dimension,
            normalize=args.normalize,
        )
        diagrams.append(dgm)
        stats = persistence_statistics(dgm)
        stats_list.append(stats)

    print("Done!")

    # Plot example persistence diagrams
    print("\nPlotting example persistence diagrams...")
    plot_example_persistence_diagrams(
        models, diagrams, output_dir / "example_persistence_diagrams.png"
    )

    # Plot topological features
    print("\nPlotting topological features...")
    plot_topological_features(
        models, stats_list, output_dir / "topological_features.png"
    )

    # Compute distance matrix
    print(f"\nComputing persistence {args.distance_metric} distance matrix...")
    distance_matrix = create_persistence_distance_matrix(diagrams, args.distance_metric)

    # Save distance matrix
    np.save(
        output_dir / f"persistence_{args.distance_metric}_distances.npy",
        distance_matrix,
    )
    print(f"Saved distance matrix")

    # Compute embedding
    print(f"\nComputing {args.embedding_method.upper()} embedding...")
    if args.embedding_method == "mds":
        embedding = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        X = embedding.fit_transform(distance_matrix)
    elif args.embedding_method == "tsne":
        embedding = TSNE(n_components=2, metric="precomputed", random_state=42)
        X = embedding.fit_transform(distance_matrix)
    else:  # PCA
        embedding = PCA(n_components=2)
        X = embedding.fit_transform(distance_matrix)

    # Determine color values
    if args.color_by == "none":
        color_values = None
        color_label = None
    elif args.color_by == "accuracy":
        color_values = np.array([m["accuracy"] for m in models])
        color_label = "Accuracy"
    elif args.color_by == "h0_births":
        color_values = np.array([s["h0_mean_birth"] for s in stats_list])
        color_label = "H0 Mean Birth"
    elif args.color_by == "h1_deaths":
        color_values = np.array([s.get("h1_mean_death", 0) for s in stats_list])
        color_label = "H1 Mean Death"

    # Plot embedding
    print(f"\nPlotting {args.embedding_method.upper()} embedding...")
    plot_persistence_embedding(
        X,
        models,
        method=args.embedding_method.upper(),
        color_values=color_values,
        color_label=color_label,
        output_path=output_dir / f"persistence_{args.embedding_method}_embedding.png",
    )

    # Save statistics
    print("\nSaving statistics summary...")
    save_statistics_summary(
        models, stats_list, output_dir / "persistence_statistics.json"
    )

    print(f"\n{'=' * 70}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nResults saved to: {output_dir}/")
    print("\nKey outputs:")
    print(f"  - example_persistence_diagrams.png: Sample persistence diagrams")
    print(f"  - topological_features.png: Feature distributions")
    print(f"  - persistence_{args.embedding_method}_embedding.png: 2D embedding")
    print(f"  - persistence_{args.distance_metric}_distances.npy: Distance matrix")
    print(f"  - persistence_statistics.json: Detailed statistics")


if __name__ == "__main__":
    main()
