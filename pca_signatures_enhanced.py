"""
Enhanced signature analysis with multiple representation methods.
Tries different ways to capture model differences that might reveal structure.
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enhanced model signature analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--summary_file",
        type=str,
        required=True,
        help="Path to evaluation_summary.json file",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file path for the plot",
    )

    parser.add_argument(
        "--signature_type",
        type=str,
        choices=[
            "histogram",  # Original histogram concatenation
            "moments",  # Statistical moments
            "percentiles",  # Percentile-based
            "wasserstein",  # Wasserstein distance matrix
            "raw_subset",  # Raw eigenvalues (subset)
            "combined",  # Combination of multiple features
        ],
        default="combined",
        help="Type of signature to use (default: combined)",
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

    parser.add_argument(
        "--annotate",
        type=str,
        choices=["none", "all", "outliers"],
        default="outliers",
        help="Which points to annotate (default: outliers)",
    )

    parser.add_argument(
        "--color_by",
        type=str,
        choices=["none", "accuracy", "frob_error", "spectral_error"],
        default="accuracy",
        help="Metric to use for coloring points (default: accuracy)",
    )

    return parser.parse_args()


def compute_moments(eigenvalues):
    """Compute statistical moments of eigenvalue distribution."""
    # Separate real and imaginary parts
    real = eigenvalues.real
    imag = eigenvalues.imag
    mag = np.abs(eigenvalues)
    angle = np.angle(eigenvalues)

    features = []

    # For each component, compute moments
    for vals in [real, imag, mag]:
        features.extend(
            [
                np.mean(vals),
                np.std(vals),
                np.median(vals),
                np.percentile(vals, 25),
                np.percentile(vals, 75),
            ]
        )

    # Angle statistics (circular statistics)
    features.extend(
        [
            np.mean(np.cos(angle)),  # Mean direction
            np.mean(np.sin(angle)),
            np.std(angle),
        ]
    )

    # Additional features
    features.extend(
        [
            np.max(mag),  # Spectral radius
            np.sum(mag > 1.0),  # Count of unstable eigenvalues
            np.sum(np.abs(real) < 0.01),  # Near-zero real parts
        ]
    )

    return np.array(features)


def compute_percentiles(eigenvalues):
    """Compute percentile-based features."""
    mag = np.abs(eigenvalues)
    angle = np.abs(np.angle(eigenvalues))

    percentiles = [0, 5, 10, 25, 50, 75, 90, 95, 100]

    features = []
    for vals in [mag, angle]:
        for p in percentiles:
            features.append(np.percentile(vals, p))

    return np.array(features)


def load_raw_eigenvalues(summary_file, seq_length):
    """Load raw eigenvalues from evaluation results."""
    # This would require storing raw eigenvalues in the summary
    # For now, return None and use histograms as fallback
    return None


def extract_signatures(summary, seq_length, signature_type):
    """Extract different types of signatures from summary data."""
    seq_key = f"seq{seq_length}"
    angle_key = f"angle_histogram_rnn_{seq_key}"
    magnitude_key = f"magnitude_histogram_rnn_{seq_key}"

    signatures = []
    model_names = []
    metrics = {"accuracy": [], "frob_error": [], "spectral_error": []}

    # Get histogram bins for reconstruction if needed
    if len(summary["metrics"]) > 0:
        first_model = summary["metrics"][0]
        if "histogram_bins" in first_model:
            angle_bins = np.array(first_model["histogram_bins"]["angle"])
            mag_bins = np.array(first_model["histogram_bins"]["magnitude"])
        else:
            # Default bins
            angle_bins = np.arange(0, np.pi + np.pi / 36, np.pi / 36)
            mag_bins = np.arange(0, 2.0 + 0.1, 0.1)

    for model in summary["metrics"]:
        if angle_key not in model or magnitude_key not in model:
            continue

        angle_hist = np.array(model[angle_key])
        mag_hist = np.array(model[magnitude_key])

        # Extract signature based on type
        if signature_type == "histogram":
            sig = np.concatenate([angle_hist, mag_hist])

        elif signature_type == "moments":
            # Reconstruct approximate eigenvalues from histograms
            # This is approximate but gives us something to work with
            angles_sampled = []
            mags_sampled = []

            # Sample from histograms
            for i, count in enumerate(angle_hist):
                if count > 0:
                    bin_center = (angle_bins[i] + angle_bins[i + 1]) / 2
                    angles_sampled.extend([bin_center] * int(count))

            for i, count in enumerate(mag_hist):
                if count > 0:
                    bin_center = (mag_bins[i] + mag_bins[i + 1]) / 2
                    mags_sampled.extend([bin_center] * int(count))

            # Create approximate eigenvalues
            n_samples = min(len(angles_sampled), len(mags_sampled))
            if n_samples > 0:
                eigenvalues = mags_sampled[:n_samples] * np.exp(
                    1j * np.array(angles_sampled[:n_samples])
                )
                sig = compute_moments(eigenvalues)
            else:
                sig = np.zeros(17)  # Default feature size

        elif signature_type == "percentiles":
            # Similar reconstruction for percentiles
            angles_sampled = []
            mags_sampled = []

            for i, count in enumerate(angle_hist):
                if count > 0:
                    bin_center = (angle_bins[i] + angle_bins[i + 1]) / 2
                    angles_sampled.extend([bin_center] * int(count))

            for i, count in enumerate(mag_hist):
                if count > 0:
                    bin_center = (mag_bins[i] + mag_bins[i + 1]) / 2
                    mags_sampled.extend([bin_center] * int(count))

            n_samples = min(len(angles_sampled), len(mags_sampled))
            if n_samples > 0:
                eigenvalues = mags_sampled[:n_samples] * np.exp(
                    1j * np.array(angles_sampled[:n_samples])
                )
                sig = compute_percentiles(eigenvalues)
            else:
                sig = np.zeros(18)

        elif signature_type == "combined":
            # Combine histogram with additional features
            hist_sig = np.concatenate([angle_hist, mag_hist])

            # Add summary statistics
            angle_mean = (
                np.sum(angle_hist * (angle_bins[:-1] + angle_bins[1:]) / 2)
                / np.sum(angle_hist)
                if np.sum(angle_hist) > 0
                else 0
            )
            angle_std = (
                np.sqrt(
                    np.sum(
                        angle_hist
                        * ((angle_bins[:-1] + angle_bins[1:]) / 2 - angle_mean) ** 2
                    )
                    / np.sum(angle_hist)
                )
                if np.sum(angle_hist) > 0
                else 0
            )

            mag_mean = (
                np.sum(mag_hist * (mag_bins[:-1] + mag_bins[1:]) / 2) / np.sum(mag_hist)
                if np.sum(mag_hist) > 0
                else 0
            )
            mag_std = (
                np.sqrt(
                    np.sum(
                        mag_hist * ((mag_bins[:-1] + mag_bins[1:]) / 2 - mag_mean) ** 2
                    )
                    / np.sum(mag_hist)
                )
                if np.sum(mag_hist) > 0
                else 0
            )

            # Find peak locations
            angle_peak = np.argmax(angle_hist)
            mag_peak = np.argmax(mag_hist)

            extra_features = np.array(
                [
                    angle_mean,
                    angle_std,
                    angle_peak,
                    mag_mean,
                    mag_std,
                    mag_peak,
                    np.sum(mag_hist[mag_bins[:-1] > 1.0]),  # Count above unit circle
                ]
            )

            sig = np.concatenate([hist_sig, extra_features])

        else:  # histogram fallback
            sig = np.concatenate([angle_hist, mag_hist])

        signatures.append(sig)
        model_names.append(model["model_name"])

        # Collect metrics for coloring
        metrics["accuracy"].append(model.get("accuracy", np.nan))
        frob_key = f"frob_relative_{seq_key}"
        spec_key = f"spectral_error_deg_{seq_key}"
        metrics["frob_error"].append(model.get(frob_key, np.nan))
        metrics["spectral_error"].append(model.get(spec_key, np.nan))

    return np.array(signatures), model_names, metrics


def identify_outliers(X, threshold=2.0):
    """Identify outliers based on distance from center."""
    center = np.mean(X, axis=0)
    distances = np.linalg.norm(X - center, axis=1)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    outlier_threshold = mean_dist + threshold * std_dist
    return distances > outlier_threshold


def main():
    args = parse_args()

    print(f"Loading data from: {args.summary_file}")
    with open(args.summary_file) as f:
        summary = json.load(f)

    print(f"Extracting signatures using method: {args.signature_type}")
    signatures, model_names, metrics = extract_signatures(
        summary, args.seq_length, args.signature_type
    )

    n_models = len(signatures)
    print(f"Loaded {n_models} models")
    print(f"Signature dimension: {signatures.shape[1]}")

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

    # Determine annotation and coloring
    if args.annotate == "none":
        annotate_mask = np.zeros(n_models, dtype=bool)
    elif args.annotate == "all":
        annotate_mask = np.ones(n_models, dtype=bool)
    else:  # outliers
        annotate_mask = identify_outliers(X, threshold=2.0)
        n_outliers = np.sum(annotate_mask)
        print(f"\nFound {n_outliers} outliers")

    # Get color values
    color_values = None
    if args.color_by != "none":
        color_values = metrics[args.color_by]
        color_values = np.array(color_values)
        valid_mask = ~np.isnan(color_values)
        print(
            f"\nColoring by {args.color_by}: {np.sum(valid_mask)}/{len(color_values)} valid values"
        )

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 9))

    if color_values is not None and np.sum(~np.isnan(color_values)) > 0:
        valid = ~np.isnan(color_values)
        scatter = ax.scatter(
            X[valid, 0],
            X[valid, 1],
            c=color_values[valid],
            cmap="viridis",
            s=60,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )
        if np.sum(~valid) > 0:
            ax.scatter(
                X[~valid, 0],
                X[~valid, 1],
                color="gray",
                s=60,
                alpha=0.3,
                label="No data",
            )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(args.color_by.replace("_", " ").title())
    else:
        ax.scatter(
            X[:, 0],
            X[:, 1],
            alpha=0.6,
            s=60,
            color="C0",
            edgecolors="black",
            linewidths=0.5,
        )

    # Add annotations
    n_annotated = 0
    for i, (x, y) in enumerate(X):
        if annotate_mask[i]:
            model_num = (
                model_names[i].split("-")[-1] if "-" in model_names[i] else str(i)
            )
            ax.annotate(
                model_num,
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="yellow",
                    alpha=0.7,
                    edgecolor="black",
                ),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=0.8),
            )
            n_annotated += 1

    # Set labels and title
    if args.method == "pca":
        ax.set_xlabel(f"PC1 ({variance_explained[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({variance_explained[1]:.1%} variance)")
        title = f"PCA: {args.signature_type} signature (seq_length={args.seq_length})"
    else:
        ax.set_xlabel("t-SNE dimension 1")
        ax.set_ylabel("t-SNE dimension 2")
        title = f"t-SNE: {args.signature_type} signature (seq_length={args.seq_length})"

    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.output_file, dpi=300, bbox_inches="tight")

    print(f"\nPlot saved to: {args.output_file}")
    print(f"Annotated {n_annotated}/{n_models} models")


if __name__ == "__main__":
    main()
