#!/usr/bin/env python3
"""
Unified Analysis Script for EMT Variable Binding Experiments.

This script provides a command-line interface for various analysis tasks:
- eigenvalue: Analyze eigenvalue distributions
- embedding: Dimensionality reduction and embedding
- persistence: Persistent homology analysis
- compare: Compare multiple model sets

Usage:
    python analyze.py eigenvalue --summary_file results.json
    python analyze.py embedding --summary_file results.json --method pca
    python analyze.py persistence --summary_file results.json
    python analyze.py compare --summary_files file1.json file2.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis import (
    dimensionality_reduction,
    distance_metrics,
    eigenvalue_analysis,
    visualization,
)


def create_parser():
    """Create argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Unified analysis tools for RNN eigenvalue distributions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Analysis command")

    # Eigenvalue analysis
    eigen_parser = subparsers.add_parser(
        "eigenvalue", help="Analyze eigenvalue distributions"
    )
    eigen_parser.add_argument(
        "--summary_file",
        type=str,
        required=True,
        help="Path to evaluation_summary.json",
    )
    eigen_parser.add_argument(
        "--output_dir", type=str, default="eigenvalue_analysis", help="Output directory"
    )
    eigen_parser.add_argument(
        "--seq_length", type=int, default=8, help="Sequence length to analyze"
    )

    # Embedding analysis
    embed_parser = subparsers.add_parser(
        "embedding", help="Dimensionality reduction and embedding"
    )
    embed_parser.add_argument(
        "--summary_file",
        type=str,
        required=True,
        help="Path to evaluation_summary.json",
    )
    embed_parser.add_argument(
        "--output_dir", type=str, default="embedding_analysis", help="Output directory"
    )
    embed_parser.add_argument(
        "--seq_length", type=int, default=8, help="Sequence length to analyze"
    )
    embed_parser.add_argument(
        "--method",
        type=str,
        default="pca",
        choices=["pca", "tsne", "mds", "isomap"],
        help="Dimensionality reduction method",
    )
    embed_parser.add_argument(
        "--signature_type",
        type=str,
        default="combined",
        choices=["histogram", "statistics", "combined"],
        help="Type of signature to extract",
    )
    embed_parser.add_argument(
        "--distance_metric",
        type=str,
        default="euclidean",
        help="Distance metric for signatures",
    )
    embed_parser.add_argument(
        "--color_by",
        type=str,
        default="accuracy",
        choices=["none", "accuracy", "frob_error", "spectral_error"],
        help="Metric for coloring points",
    )
    embed_parser.add_argument(
        "--annotate_outliers", action="store_true", help="Annotate outlier points"
    )

    # Persistence analysis
    persist_parser = subparsers.add_parser(
        "persistence", help="Persistent homology analysis"
    )
    persist_parser.add_argument(
        "--summary_file",
        type=str,
        required=True,
        help="Path to evaluation_summary.json",
    )
    persist_parser.add_argument(
        "--output_dir",
        type=str,
        default="persistence_analysis",
        help="Output directory",
    )
    persist_parser.add_argument(
        "--seq_length", type=int, default=8, help="Sequence length to analyze"
    )
    persist_parser.add_argument(
        "--max_dimension", type=int, default=1, help="Maximum homology dimension"
    )
    persist_parser.add_argument(
        "--distance_metric",
        type=str,
        default="bottleneck",
        choices=["bottleneck", "wasserstein"],
        help="Persistence distance metric",
    )
    persist_parser.add_argument(
        "--embedding_method",
        type=str,
        default="mds",
        choices=["mds", "tsne", "pca"],
        help="Method for embedding distance matrix",
    )

    # Comparison analysis
    compare_parser = subparsers.add_parser(
        "compare", help="Compare multiple model sets"
    )
    compare_parser.add_argument(
        "--summary_files",
        type=str,
        nargs="+",
        required=True,
        help="Paths to evaluation_summary.json files",
    )
    compare_parser.add_argument(
        "--labels", type=str, nargs="+", help="Labels for each model set"
    )
    compare_parser.add_argument(
        "--output_dir", type=str, default="comparison_analysis", help="Output directory"
    )
    compare_parser.add_argument(
        "--seq_length", type=int, default=8, help="Sequence length to analyze"
    )

    return parser


def eigenvalue_command(args):
    """Execute eigenvalue analysis command."""
    print(f"\n{'=' * 70}")
    print("EIGENVALUE DISTRIBUTION ANALYSIS")
    print(f"{'=' * 70}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load data
    print(f"Loading data from: {args.summary_file}")
    models = eigenvalue_analysis.load_eigenvalues_from_summary(
        args.summary_file, args.seq_length
    )
    print(f"Loaded {len(models)} models")

    # Compute statistics for each model
    print("\nComputing eigenvalue statistics...")
    all_stats = []
    for model in models:
        if "eigenvalues" in model:
            stats = eigenvalue_analysis.compute_eigenvalue_statistics(
                model["eigenvalues"]
            )
            all_stats.append(stats)

    # Save statistics (convert numpy types to native Python)
    def convert_to_native(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    stats_output = {
        "models": [
            {"model_name": model["model_name"], "statistics": convert_to_native(stats)}
            for model, stats in zip(models, all_stats)
        ]
    }

    with open(output_dir / "eigenvalue_statistics.json", "w") as f:
        json.dump(stats_output, f, indent=2)

    print(f"Saved statistics to: {output_dir / 'eigenvalue_statistics.json'}")

    # Plot eigenvalues in complex plane (first 5 models as examples)
    if any("eigenvalues" in m for m in models):
        print("\nPlotting eigenvalues in complex plane...")
        evals_list = [m["eigenvalues"] for m in models[:5] if "eigenvalues" in m]
        labels = [m["model_name"] for m in models[:5] if "eigenvalues" in m]

        visualization.plot_eigenvalues_complex_plane(
            evals_list,
            labels=labels,
            title="Eigenvalues in Complex Plane (First 5 Models)",
            output_path=output_dir / "complex_plane_examples.png",
        )

    # Create scatter plot matrix of statistics
    if all_stats:
        print("\nCreating scatter plot matrix...")
        stat_names = ["spectral_radius", "mag_mean", "mag_std", "n_unstable"]
        stat_data = {name: [s[name] for s in all_stats] for name in stat_names}

        visualization.plot_scatter_matrix(
            stat_data, output_path=output_dir / "statistics_scatter_matrix.png"
        )

    print(f"\n{'=' * 70}")
    print(f"Analysis complete! Results in: {output_dir}/")
    print(f"{'=' * 70}\n")


def embedding_command(args):
    """Execute embedding analysis command."""
    print(f"\n{'=' * 70}")
    print("EMBEDDING ANALYSIS")
    print(f"{'=' * 70}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load data
    print(f"Loading data from: {args.summary_file}")
    models = eigenvalue_analysis.load_eigenvalues_from_summary(
        args.summary_file, args.seq_length
    )
    print(f"Loaded {len(models)} models")

    # Extract signatures
    print(f"\nExtracting signatures (type: {args.signature_type})...")
    signatures, model_names, metrics = eigenvalue_analysis.extract_signatures(
        models, signature_type=args.signature_type
    )
    print(f"Signature shape: {signatures.shape}")

    # Compute dimensionality reduction
    print(f"\nComputing {args.method.upper()} embedding...")
    X, info = dimensionality_reduction.compute_dimensionality_reduction(
        signatures, method=args.method, n_components=2
    )

    # Save embedding
    np.save(output_dir / f"{args.method}_embedding.npy", X)

    # Plot variance if PCA
    if args.method == "pca" and "explained_variance" in info:
        print(f"Explained variance: {info['explained_variance']}")
        visualization.plot_pca_variance(
            info["explained_variance"], output_path=output_dir / "pca_variance.png"
        )

    # Determine coloring
    color_values = None
    color_label = None
    if args.color_by != "none":
        color_values = np.array(metrics[args.color_by])
        color_label = args.color_by.replace("_", " ").title()

    # Identify outliers for annotation
    annotate_indices = None
    if args.annotate_outliers:
        # Compute distance matrix and find outliers
        dist_matrix = distance_metrics.compute_pairwise_distances(signatures)
        outlier_mask = distance_metrics.identify_outliers_by_distance(
            dist_matrix, threshold=2.0
        )
        annotate_indices = np.where(outlier_mask)[0]
        print(f"Found {len(annotate_indices)} outliers to annotate")

    # Create plot
    print(f"\nCreating {args.method.upper()} plot...")
    xlabel = f"{args.method.upper()} Dimension 1"
    ylabel = f"{args.method.upper()} Dimension 2"

    if args.method == "pca" and "explained_variance" in info:
        var = info["explained_variance"]
        xlabel = f"PC1 ({var[0]:.1%} variance)"
        ylabel = f"PC2 ({var[1]:.1%} variance)"

    visualization.plot_embedding_2d(
        X,
        labels=model_names,
        color_values=color_values,
        color_label=color_label,
        title=f"{args.method.upper()} Embedding: {args.signature_type} signatures",
        xlabel=xlabel,
        ylabel=ylabel,
        annotate_indices=annotate_indices,
        output_path=output_dir / f"{args.method}_embedding.png",
    )

    # Compute and save distance matrix
    print("\nComputing pairwise distances...")
    dist_matrix = distance_metrics.compute_pairwise_distances(
        signatures, metric=args.distance_metric
    )
    np.save(output_dir / "distance_matrix.npy", dist_matrix)

    # Compute distance statistics
    dist_stats = distance_metrics.compute_distance_statistics(dist_matrix)
    with open(output_dir / "distance_statistics.json", "w") as f:
        json.dump(dist_stats, f, indent=2)

    print("\nDistance statistics:")
    for key, value in dist_stats.items():
        print(f"  {key}: {value:.4f}")

    print(f"\n{'=' * 70}")
    print(f"Analysis complete! Results in: {output_dir}/")
    print(f"{'=' * 70}\n")


def persistence_command(args):
    """Execute persistent homology analysis command."""
    print(f"\n{'=' * 70}")
    print("PERSISTENT HOMOLOGY ANALYSIS")
    print(f"{'=' * 70}\n")

    # Import persistent homology module
    try:
        from tqdm import tqdm

        from analysis import persistent_homology
    except ImportError as e:
        print(f"ERROR: Could not import required module: {e}")
        print(
            "For persistent homology, install with: pip install ripser persim scikit-tda tqdm"
        )
        return

    if not persistent_homology.RIPSER_AVAILABLE:
        print("\nERROR: This command requires ripser and persim.")
        print("Install with: pip install ripser persim scikit-tda")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Loading eigenvalues from: {args.summary_file}")
    models = persistent_homology.load_eigenvalues_from_summary(
        args.summary_file, args.seq_length
    )
    print(f"Loaded {len(models)} models")

    # Compute persistence diagrams
    print(f"\nComputing persistence diagrams (max dimension = {args.max_dimension})...")
    diagrams = []
    stats_list = []

    for model in tqdm(models):
        dgm = persistent_homology.compute_persistence_diagram(
            model["eigenvalues"], max_dimension=args.max_dimension, normalize=False
        )
        diagrams.append(dgm)
        stats = persistent_homology.persistence_statistics(dgm)
        stats_list.append(stats)

    print("Done!")

    # Plot example persistence diagrams
    print("\nPlotting example persistence diagrams...")
    persistent_homology.plot_example_persistence_diagrams(
        models, diagrams, output_dir / "example_persistence_diagrams.png"
    )

    # Plot topological features
    print("\nPlotting topological features...")
    persistent_homology.plot_topological_features(
        models, stats_list, output_dir / "topological_features.png"
    )

    # Compute distance matrix
    print(f"\nComputing persistence {args.distance_metric} distance matrix...")
    distance_matrix = persistent_homology.create_persistence_distance_matrix(
        diagrams, args.distance_metric
    )

    # Save distance matrix
    np.save(
        output_dir / f"persistence_{args.distance_metric}_distances.npy",
        distance_matrix,
    )
    print(f"Saved distance matrix")

    # Compute embedding
    print(f"\nComputing {args.embedding_method.upper()} embedding...")
    from analysis.dimensionality_reduction import apply_mds, apply_pca, apply_tsne

    if args.embedding_method == "mds":
        X = apply_mds(distance_matrix, n_components=2, dissimilarity="precomputed")
    elif args.embedding_method == "tsne":
        X = apply_tsne(distance_matrix, n_components=2, metric="precomputed")
    else:  # PCA
        X = apply_pca(distance_matrix, n_components=2)[0]

    # Determine color values
    color_values = np.array([m["accuracy"] for m in models])
    color_label = "Accuracy"

    # Plot embedding
    print(f"\nPlotting {args.embedding_method.upper()} embedding...")
    persistent_homology.plot_persistence_embedding(
        X,
        models,
        method=args.embedding_method.upper(),
        color_values=color_values,
        color_label=color_label,
        output_path=output_dir / f"persistence_{args.embedding_method}_embedding.png",
    )

    # Save statistics
    print("\nSaving statistics summary...")
    persistent_homology.save_statistics_summary(
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


def compare_command(args):
    """Execute comparison analysis command."""
    print(f"\n{'=' * 70}")
    print("COMPARISON ANALYSIS")
    print(f"{'=' * 70}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load all model sets
    all_models = []
    all_labels = []

    for i, summary_file in enumerate(args.summary_files):
        label = (
            args.labels[i] if args.labels and i < len(args.labels) else f"Set {i + 1}"
        )
        print(f"\nLoading {label} from: {summary_file}")

        models = eigenvalue_analysis.load_eigenvalues_from_summary(
            summary_file, args.seq_length
        )
        all_models.append(models)
        all_labels.append(label)
        print(f"  Loaded {len(models)} models")

    # Compare eigenvalue distributions
    print("\nComparing eigenvalue distributions...")

    # Collect all eigenvalues for plotting
    all_evals = []
    plot_labels = []

    for models, label in zip(all_models, all_labels):
        # Take first model from each set as representative
        if models and "eigenvalues" in models[0]:
            all_evals.append(models[0]["eigenvalues"])
            plot_labels.append(f"{label} (example)")

    if all_evals:
        visualization.plot_eigenvalues_complex_plane(
            all_evals,
            labels=plot_labels,
            title="Eigenvalue Comparison (Example Models)",
            output_path=output_dir / "comparison_complex_plane.png",
        )

    # Compare statistics
    print("\nComputing statistics for each set...")
    set_stats = []

    for models, label in zip(all_models, all_labels):
        stats_list = []
        for model in models:
            if "eigenvalues" in model:
                stats = eigenvalue_analysis.compute_eigenvalue_statistics(
                    model["eigenvalues"]
                )
                stats_list.append(stats)

        if stats_list:
            # Aggregate statistics
            agg_stats = {}
            for key in stats_list[0].keys():
                values = [s[key] for s in stats_list]
                agg_stats[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

            set_stats.append(
                {"label": label, "n_models": len(models), "statistics": agg_stats}
            )

    # Save comparison results
    with open(output_dir / "comparison_statistics.json", "w") as f:
        json.dump(set_stats, f, indent=2)

    print("\nComparison statistics saved")

    # Create comparison plots
    if set_stats:
        print("\nCreating comparison plots...")

        metrics = ["spectral_radius", "mag_mean", "n_unstable"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))

        for i, metric in enumerate(metrics):
            ax = axes[i]
            means = [s["statistics"][metric]["mean"] for s in set_stats]
            stds = [s["statistics"][metric]["std"] for s in set_stats]
            labels = [s["label"] for s in set_stats]

            x_pos = np.arange(len(labels))
            ax.bar(x_pos, means, yerr=stds, alpha=0.7, capsize=5, edgecolor="black")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f"{metric.replace('_', ' ').title()} Comparison")
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(output_dir / "comparison_metrics.png", dpi=300, bbox_inches="tight")
        plt.close()

    print(f"\n{'=' * 70}")
    print(f"Comparison complete! Results in: {output_dir}/")
    print(f"{'=' * 70}\n")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == "eigenvalue":
        eigenvalue_command(args)
    elif args.command == "embedding":
        embedding_command(args)
    elif args.command == "persistence":
        persistence_command(args)
    elif args.command == "compare":
        compare_command(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
