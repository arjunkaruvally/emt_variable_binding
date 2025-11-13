import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize model signatures using PCA/t-SNE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Annotate all points
  python pca_signatures.py --summary_file data/evaluation_summary.json --output_file pca.png --annotate all

  # Annotate outliers only (points beyond N std devs from center)
  python pca_signatures.py --summary_file data/evaluation_summary.json --output_file pca.png --annotate outliers --outlier_threshold 2.0

  # Annotate specific model indices
  python pca_signatures.py --summary_file data/evaluation_summary.json --output_file pca.png --annotate specific --indices 5 12 23
        """,
    )

    parser.add_argument(
        "--summary_file",
        type=str,
        default="50_models_seqlen_8/evaluation_summary.json",
        help="Path to evaluation_summary.json file",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file path for the plot",
    )

    parser.add_argument(
        "--annotate",
        type=str,
        choices=["none", "all", "outliers", "specific"],
        default="all",
        help="Which points to annotate (default: all)",
    )

    parser.add_argument(
        "--outlier_threshold",
        type=float,
        default=2.0,
        help="Number of standard deviations for outlier detection (default: 2.0)",
    )

    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        help="Specific model indices to annotate (0-based)",
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["pca", "tsne"],
        default="pca",
        help="Dimensionality reduction method (default: pca)",
    )

    parser.add_argument(
        "--seq_length",
        type=int,
        default=8,
        help="Sequence length to analyze (default: 8)",
    )

    return parser.parse_args()


def identify_outliers(X, threshold=2.0):
    """
    Identify outliers based on distance from center.

    Args:
        X: 2D array of shape (n_samples, 2)
        threshold: Number of standard deviations for outlier detection

    Returns:
        Array of boolean flags indicating outliers
    """
    # Calculate distance from center (mean)
    center = np.mean(X, axis=0)
    distances = np.linalg.norm(X - center, axis=1)

    # Calculate threshold based on standard deviation
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    outlier_threshold = mean_dist + threshold * std_dist

    return distances > outlier_threshold


if __name__ == "__main__":
    args = parse_args()

    # Load evaluation summary
    print(f"Loading data from: {args.summary_file}")
    with open(args.summary_file) as f:
        summary = json.load(f)

    # Extract signatures for specified sequence length
    seq_key = f"seq{args.seq_length}"
    angle_key = f"angle_histogram_rnn_{seq_key}"
    magnitude_key = f"magnitude_histogram_rnn_{seq_key}"

    signatures = []
    model_names = []
    for model in summary["metrics"]:
        if angle_key in model and magnitude_key in model:
            signatures.append(model[angle_key] + model[magnitude_key])
            model_names.append(model["model_name"])
        else:
            print(
                f"Warning: Missing histogram data for {model.get('model_name', 'unknown')}"
            )

    signatures = np.array(signatures)
    n_models = len(signatures)

    print(f"Loaded {n_models} model signatures")
    print(f"Signature shape: {signatures.shape}")

    # Perform dimensionality reduction
    if args.method == "pca":
        print("Using PCA for dimensionality reduction")
        reduction = PCA(n_components=2)
    else:
        print("Using t-SNE for dimensionality reduction")
        reduction = TSNE(n_components=2, random_state=42)

    X = reduction.fit_transform(signatures)
    print(f"Reduced data shape: {X.shape}")

    if args.method == "pca":
        variance_explained = reduction.explained_variance_ratio_
        print(f"Variance explained by PC1: {variance_explained[0]:.2%}")
        print(f"Variance explained by PC2: {variance_explained[1]:.2%}")
        print(f"Total variance explained: {sum(variance_explained):.2%}")

    # Determine which points to annotate
    if args.annotate == "none":
        annotate_mask = np.zeros(n_models, dtype=bool)
    elif args.annotate == "all":
        annotate_mask = np.ones(n_models, dtype=bool)
    elif args.annotate == "outliers":
        annotate_mask = identify_outliers(X, threshold=args.outlier_threshold)
        n_outliers = np.sum(annotate_mask)
        outlier_indices = np.where(annotate_mask)[0]
        print(
            f"\nFound {n_outliers} outliers (threshold={args.outlier_threshold} std):"
        )
        for idx in outlier_indices:
            print(f"  Model {idx}: {model_names[idx]}")
    elif args.annotate == "specific":
        if args.indices is None:
            print("Error: --indices required when using --annotate specific")
            exit(1)
        annotate_mask = np.zeros(n_models, dtype=bool)
        for idx in args.indices:
            if 0 <= idx < n_models:
                annotate_mask[idx] = True
            else:
                print(f"Warning: Index {idx} out of range [0, {n_models - 1}]")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot with different colors for annotated vs non-annotated
    if args.annotate != "none":
        # Plot non-annotated points
        ax.scatter(
            X[~annotate_mask, 0],
            X[~annotate_mask, 1],
            alpha=0.6,
            s=50,
            color="C0",
            label="Models",
        )
        # Plot annotated points with different color
        ax.scatter(
            X[annotate_mask, 0],
            X[annotate_mask, 1],
            alpha=0.8,
            s=80,
            color="red",
            marker="o",
            edgecolors="black",
            linewidths=1.5,
            label="Annotated" if args.annotate == "outliers" else None,
        )
    else:
        ax.scatter(X[:, 0], X[:, 1], alpha=0.6, s=50, color="C0")

    # Add annotations
    n_annotated = 0
    for i, (x, y) in enumerate(X):
        if annotate_mask[i]:
            # Extract model number from model name (e.g., "model-5" -> 5)
            model_num = (
                model_names[i].split("-")[-1] if "-" in model_names[i] else str(i)
            )
            ax.annotate(
                model_num,
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="yellow",
                    alpha=0.7,
                    edgecolor="black",
                ),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=1),
            )
            n_annotated += 1

    # Set labels and title
    if args.method == "pca":
        ax.set_xlabel(f"PC1 ({variance_explained[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({variance_explained[1]:.1%} variance)")
        title = f"PCA of Model Signatures (seq_length={args.seq_length})"
    else:
        ax.set_xlabel("t-SNE dimension 1")
        ax.set_ylabel("t-SNE dimension 2")
        title = f"t-SNE of Model Signatures (seq_length={args.seq_length})"

    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if args.annotate != "none":
        ax.legend()

    fig.tight_layout()
    fig.savefig(args.output_file, dpi=300, bbox_inches="tight")

    print(f"\nPlot saved to: {args.output_file}")
    print(f"Annotated {n_annotated}/{n_models} models")
