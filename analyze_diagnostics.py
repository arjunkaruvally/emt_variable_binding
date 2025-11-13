"""
Numerical analysis of diagnostic results to understand lack of structure.
Generates a text report with key insights.
"""

import argparse
import json

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_file", type=str, required=True)
    parser.add_argument("--seq_length", type=int, default=8)
    return parser.parse_args()


def analyze_variance(signatures, angle_hists, magnitude_hists):
    """Analyze variance in different components."""
    print("\n" + "=" * 70)
    print("VARIANCE ANALYSIS")
    print("=" * 70)

    # Total variance
    total_var = np.var(signatures, axis=0)
    print(f"\nTotal signature variance statistics:")
    print(f"  Mean variance: {np.mean(total_var):.6f}")
    print(f"  Median variance: {np.median(total_var):.6f}")
    print(f"  Max variance: {np.max(total_var):.6f}")
    print(f"  Min variance: {np.min(total_var):.6f}")

    # Coefficient of variation (std/mean) for each bin
    means = np.mean(signatures, axis=0)
    stds = np.std(signatures, axis=0)
    cv = np.abs(stds / (means + 1e-10))  # Avoid division by zero

    print(f"\nCoefficient of variation (CV = std/mean):")
    print(f"  Mean CV: {np.mean(cv):.4f}")
    print(f"  Median CV: {np.median(cv):.4f}")
    print(f"  Bins with CV > 0.1: {np.sum(cv > 0.1)} / {len(cv)}")
    print(f"  Bins with CV > 0.2: {np.sum(cv > 0.2)} / {len(cv)}")

    # Angle vs magnitude variance
    angle_var = np.var(angle_hists, axis=0)
    mag_var = np.var(magnitude_hists, axis=0)

    print(f"\nAngle histogram variance:")
    print(f"  Mean: {np.mean(angle_var):.6f}")
    print(f"  Total: {np.sum(angle_var):.6f}")

    print(f"\nMagnitude histogram variance:")
    print(f"  Mean: {np.mean(mag_var):.6f}")
    print(f"  Total: {np.sum(mag_var):.6f}")

    # Which contributes more?
    ratio = np.sum(mag_var) / (np.sum(angle_var) + 1e-10)
    print(f"\nVariance ratio (magnitude/angle): {ratio:.2f}")
    if ratio > 2:
        print("  → Magnitude varies much more than angle")
    elif ratio > 0.5:
        print("  → Magnitude and angle vary comparably")
    else:
        print("  → Angle varies much more than magnitude")

    return {
        "total_var_mean": np.mean(total_var),
        "cv_mean": np.mean(cv),
        "angle_var_total": np.sum(angle_var),
        "mag_var_total": np.sum(mag_var),
    }


def analyze_distances(signatures):
    """Analyze pairwise distances between signatures."""
    print("\n" + "=" * 70)
    print("PAIRWISE DISTANCE ANALYSIS")
    print("=" * 70)

    distances = pdist(signatures, metric="euclidean")

    print(f"\nPairwise distance statistics:")
    print(f"  Mean: {np.mean(distances):.4f}")
    print(f"  Median: {np.median(distances):.4f}")
    print(f"  Std: {np.std(distances):.4f}")
    print(f"  Min: {np.min(distances):.4f}")
    print(f"  Max: {np.max(distances):.4f}")

    # Relative spread
    relative_spread = np.std(distances) / np.mean(distances)
    print(f"\nRelative spread (std/mean): {relative_spread:.4f}")

    if relative_spread < 0.1:
        print("  → Very tight clustering - models are extremely similar")
    elif relative_spread < 0.2:
        print("  → Tight clustering - models are very similar")
    elif relative_spread < 0.3:
        print("  → Moderate clustering - some variation but mostly similar")
    else:
        print("  → Good spread - models show meaningful differences")

    # Check if distances are normally distributed
    # (which would suggest random point cloud)
    from scipy import stats

    _, p_value = stats.normaltest(distances)
    print(f"\nNormality test (p-value): {p_value:.4f}")
    if p_value > 0.05:
        print("  → Distances appear normally distributed (suggests random cloud)")
    else:
        print("  → Distances deviate from normal (suggests structure)")

    return {
        "dist_mean": np.mean(distances),
        "dist_std": np.std(distances),
        "relative_spread": relative_spread,
        "normality_pval": p_value,
    }


def analyze_pca(signatures):
    """Analyze PCA decomposition."""
    print("\n" + "=" * 70)
    print("PCA ANALYSIS")
    print("=" * 70)

    pca = PCA(n_components=min(10, signatures.shape[1]))
    pca.fit(signatures)

    var_explained = pca.explained_variance_ratio_

    print(f"\nExplained variance by component:")
    for i in range(min(5, len(var_explained))):
        print(f"  PC{i + 1}: {var_explained[i]:.2%}")

    cumsum = np.cumsum(var_explained)
    print(f"\nCumulative explained variance:")
    print(f"  First 2 PCs: {cumsum[1]:.2%}")
    print(f"  First 3 PCs: {cumsum[2]:.2%}")
    print(f"  First 5 PCs: {cumsum[min(4, len(cumsum) - 1)]:.2%}")

    # Check intrinsic dimensionality
    # Count components needed for 90% variance
    n_90 = np.searchsorted(cumsum, 0.9) + 1
    n_95 = np.searchsorted(cumsum, 0.95) + 1

    print(f"\nIntrinsic dimensionality:")
    print(f"  Components for 90% variance: {n_90} / {signatures.shape[1]}")
    print(f"  Components for 95% variance: {n_95} / {signatures.shape[1]}")

    if cumsum[1] < 0.20:
        print("\n⚠ WARNING: First 2 PCs explain < 20% variance")
        print("  → PCA plot is showing only a tiny fraction of variation")
        print("  → Most differences are in higher dimensions")
        print("  → Consider using more informative features or different viz")
    elif cumsum[1] < 0.40:
        print("\n⚠ First 2 PCs explain < 40% variance")
        print("  → PCA plot shows limited view of variation")
        print("  → Significant structure may exist in higher dimensions")
    else:
        print("\n✓ First 2 PCs capture substantial variance")
        print("  → PCA plot is a good 2D representation")

    # Check if variance is concentrated or spread out
    # Using entropy of explained variance
    entropy = -np.sum(var_explained * np.log(var_explained + 1e-10))
    max_entropy = np.log(len(var_explained))
    normalized_entropy = entropy / max_entropy

    print(f"\nVariance distribution entropy: {normalized_entropy:.3f}")
    if normalized_entropy > 0.8:
        print("  → Variance is spread across many dimensions (flat spectrum)")
    elif normalized_entropy > 0.6:
        print("  → Variance is moderately distributed")
    else:
        print("  → Variance is concentrated in few dimensions")

    return {
        "pc1_var": var_explained[0],
        "pc2_var": var_explained[1],
        "pc12_cumvar": cumsum[1],
        "n_90": n_90,
        "entropy": normalized_entropy,
    }


def analyze_metrics(accuracies, frob_errors, spectral_errors):
    """Analyze performance metrics."""
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS ANALYSIS")
    print("=" * 70)

    # Accuracy
    valid_acc = accuracies[~np.isnan(accuracies)]
    print(f"\nAccuracy statistics (n={len(valid_acc)}):")
    print(f"  Mean: {np.mean(valid_acc):.6f}")
    print(f"  Std: {np.std(valid_acc):.6f}")
    print(f"  Min: {np.min(valid_acc):.6f}")
    print(f"  Max: {np.max(valid_acc):.6f}")
    print(f"  Range: {np.max(valid_acc) - np.min(valid_acc):.6f}")

    acc_cv = np.std(valid_acc) / np.mean(valid_acc)
    print(f"  CV: {acc_cv:.6f}")

    if np.std(valid_acc) < 0.001:
        print("  → Extremely tight accuracy - all models converged to same solution")
    elif np.std(valid_acc) < 0.01:
        print("  → Very tight accuracy - minimal performance variation")
    elif np.std(valid_acc) < 0.05:
        print("  → Moderate accuracy variation")
    else:
        print("  → Significant accuracy variation")

    # Frobenius errors
    valid_frob = frob_errors[~np.isnan(frob_errors)]
    if len(valid_frob) > 0:
        print(f"\nFrobenius error statistics (n={len(valid_frob)}):")
        print(f"  Mean: {np.mean(valid_frob):.6f}")
        print(f"  Std: {np.std(valid_frob):.6f}")
        print(f"  Min: {np.min(valid_frob):.6f}")
        print(f"  Max: {np.max(valid_frob):.6f}")
        print(f"  Range: {np.max(valid_frob) - np.min(valid_frob):.6f}")

    # Spectral errors
    valid_spec = spectral_errors[~np.isnan(spectral_errors)]
    if len(valid_spec) > 0:
        print(f"\nSpectral error statistics (n={len(valid_spec)}):")
        print(f"  Mean: {np.mean(valid_spec):.6f}")
        print(f"  Std: {np.std(valid_spec):.6f}")
        print(f"  Min: {np.min(valid_spec):.6f}")
        print(f"  Max: {np.max(valid_spec):.6f}")
        print(f"  Range: {np.max(valid_spec) - np.min(valid_spec):.6f}")

    return {
        "acc_mean": np.mean(valid_acc),
        "acc_std": np.std(valid_acc),
        "acc_range": np.max(valid_acc) - np.min(valid_acc),
    }


def main():
    args = parse_args()

    # Load data
    with open(args.summary_file) as f:
        summary = json.load(f)

    seq_key = f"seq{args.seq_length}"
    angle_key = f"angle_histogram_rnn_{seq_key}"
    magnitude_key = f"magnitude_histogram_rnn_{seq_key}"

    signatures = []
    angle_hists = []
    magnitude_hists = []
    accuracies = []
    frob_errors = []
    spectral_errors = []

    for model in summary["metrics"]:
        if angle_key in model and magnitude_key in model:
            signatures.append(model[angle_key] + model[magnitude_key])
            angle_hists.append(model[angle_key])
            magnitude_hists.append(model[magnitude_key])
            accuracies.append(model.get("accuracy", np.nan))
            frob_key = f"frob_relative_{seq_key}"
            spec_key = f"spectral_error_deg_{seq_key}"
            frob_errors.append(model.get(frob_key, np.nan))
            spectral_errors.append(model.get(spec_key, np.nan))

    signatures = np.array(signatures)
    angle_hists = np.array(angle_hists)
    magnitude_hists = np.array(magnitude_hists)
    accuracies = np.array(accuracies)
    frob_errors = np.array(frob_errors)
    spectral_errors = np.array(spectral_errors)

    print("=" * 70)
    print("DIAGNOSTIC SUMMARY REPORT")
    print("=" * 70)
    print(f"\nDataset: {args.summary_file}")
    print(f"Number of models: {len(signatures)}")
    print(f"Signature dimension: {signatures.shape[1]}")
    print(f"Sequence length: {args.seq_length}")

    # Run analyses
    var_results = analyze_variance(signatures, angle_hists, magnitude_hists)
    dist_results = analyze_distances(signatures)
    pca_results = analyze_pca(signatures)
    metric_results = analyze_metrics(accuracies, frob_errors, spectral_errors)

    # Final diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS: WHY NO STRUCTURE IN PCA?")
    print("=" * 70)

    diagnosis = []

    # Check 1: Low variance
    if var_results["total_var_mean"] < 1e-4:
        diagnosis.append("✗ VERY LOW VARIANCE: Models are extremely similar")
    elif var_results["cv_mean"] < 0.05:
        diagnosis.append(
            "✗ LOW COEFFICIENT OF VARIATION: Small differences relative to mean"
        )

    # Check 2: Tight distances
    if dist_results["relative_spread"] < 0.15:
        diagnosis.append("✗ TIGHT CLUSTERING: All models are very close together")

    # Check 3: PCA captures little variance
    if pca_results["pc12_cumvar"] < 0.2:
        diagnosis.append("✗ LOW PCA VARIANCE: First 2 PCs explain < 20% of variance")
        diagnosis.append("  → Variation exists but is high-dimensional")

    # Check 4: Perfect accuracy
    if metric_results["acc_std"] < 0.001:
        diagnosis.append("✗ NO PERFORMANCE VARIATION: All models achieve same accuracy")

    # Check 5: Random cloud
    if dist_results["normality_pval"] > 0.05 and dist_results["relative_spread"] > 0.15:
        diagnosis.append(
            "⚠ NORMALLY DISTRIBUTED DISTANCES: Suggests random point cloud"
        )

    if len(diagnosis) == 0:
        print("\n✓ No obvious issues - structure should be visible!")
        print("  Try adjusting visualization parameters or outlier thresholds")
    else:
        print("\nIssues identified:")
        for i, issue in enumerate(diagnosis, 1):
            print(f"\n{i}. {issue}")

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    recommendations = []

    if pca_results["pc12_cumvar"] < 0.3:
        recommendations.append(
            "• Try 3D PCA or visualize higher PCs (PC2 vs PC3, etc.)"
        )
        recommendations.append(
            "• Use t-SNE which may capture high-dimensional structure better"
        )

    if var_results["cv_mean"] < 0.1:
        recommendations.append("• Models are too similar - consider:")
        recommendations.append("  - Training with different hyperparameters")
        recommendations.append("  - Using different random seeds with larger intervals")
        recommendations.append(
            "  - Stopping training early to see learning trajectories"
        )
        recommendations.append("  - Training with different learning rates")

    if metric_results["acc_std"] < 0.01 and metric_results["acc_mean"] > 0.95:
        recommendations.append("• All models solved the task perfectly")
        recommendations.append(
            "  This is good for reliability but limits interpretability"
        )
        recommendations.append("  Consider analyzing intermediate checkpoints")

    if dist_results["relative_spread"] < 0.2:
        recommendations.append("• Try different signature representations:")
        recommendations.append("  - Statistical moments instead of histograms")
        recommendations.append("  - Raw eigenvalue positions (not histograms)")
        recommendations.append("  - Weight matrix norms or other global properties")

    if var_results["mag_var_total"] > 3 * var_results["angle_var_total"]:
        recommendations.append("• Magnitude varies more than angle")
        recommendations.append("  Focus on magnitude-based features")
    elif var_results["angle_var_total"] > 3 * var_results["mag_var_total"]:
        recommendations.append("• Angle varies more than magnitude")
        recommendations.append("  Focus on angle-based features")

    if len(recommendations) == 0:
        recommendations.append("• Data looks good - structure should be visible")
        recommendations.append(
            "• Check visualization settings (scaling, coloring, etc.)"
        )

    for rec in recommendations:
        print(rec)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
