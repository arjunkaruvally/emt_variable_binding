"""
Core eigenvalue analysis utilities.

This module provides functions for loading, processing, and analyzing
eigenvalue distributions from trained RNN models.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import wasserstein_distance


def load_eigenvalues_from_summary(summary_file: str, seq_length: int = 8) -> List[Dict]:
    """
    Load eigenvalues and metadata from evaluation summary.

    Args:
        summary_file: Path to evaluation_summary.json
        seq_length: Sequence length to extract

    Returns:
        List of dicts with model_name, eigenvalues, and metrics
    """
    with open(summary_file) as f:
        summary = json.load(f)

    seq_key = f"seq{seq_length}"
    eigenvalue_key = f"eigenvalues_rnn_{seq_key}"
    angle_key = f"angle_histogram_rnn_{seq_key}"
    magnitude_key = f"magnitude_histogram_rnn_{seq_key}"

    models = []
    for model_data in summary["metrics"]:
        model_info = {
            "model_name": model_data["model_name"],
            "accuracy": model_data.get("accuracy", np.nan),
            "frob_error": model_data.get(f"frob_relative_{seq_key}", np.nan),
            "spectral_error": model_data.get(f"spectral_error_deg_{seq_key}", np.nan),
        }

        # Load eigenvalues if available
        if eigenvalue_key in model_data:
            evals_list = model_data[eigenvalue_key]
            eigenvalues = np.array([complex(e[0], e[1]) for e in evals_list])
            model_info["eigenvalues"] = eigenvalues

        # Load histograms if available
        if angle_key in model_data:
            model_info["angle_histogram"] = np.array(model_data[angle_key])
        if magnitude_key in model_data:
            model_info["magnitude_histogram"] = np.array(model_data[magnitude_key])

        # Load histogram bins if available
        if "histogram_bins" in model_data:
            model_info["angle_bins"] = np.array(model_data["histogram_bins"]["angle"])
            model_info["magnitude_bins"] = np.array(
                model_data["histogram_bins"]["magnitude"]
            )

        models.append(model_info)

    return models


def compute_eigenvalue_statistics(eigenvalues: np.ndarray) -> Dict:
    """
    Compute statistical features of eigenvalue distribution.

    Args:
        eigenvalues: Complex array of eigenvalues

    Returns:
        Dictionary of statistical features
    """
    mag = np.abs(eigenvalues)
    angle = np.angle(eigenvalues)
    real = eigenvalues.real
    imag = eigenvalues.imag

    stats = {
        # Magnitude statistics
        "mag_mean": np.mean(mag),
        "mag_std": np.std(mag),
        "mag_median": np.median(mag),
        "mag_min": np.min(mag),
        "mag_max": np.max(mag),
        "mag_q25": np.percentile(mag, 25),
        "mag_q75": np.percentile(mag, 75),
        # Angle statistics
        "angle_mean": np.mean(angle),
        "angle_std": np.std(angle),
        "angle_median": np.median(angle),
        # Real part statistics
        "real_mean": np.mean(real),
        "real_std": np.std(real),
        # Imaginary part statistics
        "imag_mean": np.mean(imag),
        "imag_std": np.std(imag),
        # Stability metrics
        "spectral_radius": np.max(mag),
        "n_unstable": np.sum(mag > 1.0),
        "n_near_unit_circle": np.sum((mag > 0.95) & (mag < 1.05)),
        # Circular statistics
        "mean_direction_cos": np.mean(np.cos(angle)),
        "mean_direction_sin": np.mean(np.sin(angle)),
        # Spread metrics
        "mag_range": np.max(mag) - np.min(mag),
        "angle_range": np.max(angle) - np.min(angle),
    }

    return stats


def histogram_to_eigenvalues(
    angle_hist: np.ndarray,
    magnitude_hist: np.ndarray,
    angle_bins: np.ndarray,
    magnitude_bins: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct approximate eigenvalues from histograms.

    Args:
        angle_hist: Histogram of angles
        magnitude_hist: Histogram of magnitudes
        angle_bins: Bin edges for angles
        magnitude_bins: Bin edges for magnitudes

    Returns:
        Array of complex eigenvalues (approximate)
    """
    angles_sampled = []
    mags_sampled = []

    # Sample from angle histogram
    for i, count in enumerate(angle_hist):
        if count > 0:
            bin_center = (angle_bins[i] + angle_bins[i + 1]) / 2
            angles_sampled.extend([bin_center] * int(count))

    # Sample from magnitude histogram
    for i, count in enumerate(magnitude_hist):
        if count > 0:
            bin_center = (magnitude_bins[i] + magnitude_bins[i + 1]) / 2
            mags_sampled.extend([bin_center] * int(count))

    # Create complex eigenvalues
    n_samples = min(len(angles_sampled), len(mags_sampled))
    if n_samples == 0:
        return np.array([])

    eigenvalues = np.array(mags_sampled[:n_samples]) * np.exp(
        1j * np.array(angles_sampled[:n_samples])
    )

    return eigenvalues


def create_signature_from_histograms(
    angle_hist: np.ndarray,
    magnitude_hist: np.ndarray,
    angle_bins: Optional[np.ndarray] = None,
    magnitude_bins: Optional[np.ndarray] = None,
    include_statistics: bool = True,
) -> np.ndarray:
    """
    Create a feature signature from histograms.

    Args:
        angle_hist: Histogram of angles
        magnitude_hist: Histogram of magnitudes
        angle_bins: Bin edges for angles (for computing statistics)
        magnitude_bins: Bin edges for magnitudes (for computing statistics)
        include_statistics: If True, append statistical features

    Returns:
        Feature vector
    """
    signature = np.concatenate([angle_hist, magnitude_hist])

    if include_statistics and angle_bins is not None and magnitude_bins is not None:
        # Compute weighted statistics
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
            np.sum(magnitude_hist * (magnitude_bins[:-1] + magnitude_bins[1:]) / 2)
            / np.sum(magnitude_hist)
            if np.sum(magnitude_hist) > 0
            else 0
        )
        mag_std = (
            np.sqrt(
                np.sum(
                    magnitude_hist
                    * ((magnitude_bins[:-1] + magnitude_bins[1:]) / 2 - mag_mean) ** 2
                )
                / np.sum(magnitude_hist)
            )
            if np.sum(magnitude_hist) > 0
            else 0
        )

        # Peak locations
        angle_peak = np.argmax(angle_hist)
        mag_peak = np.argmax(magnitude_hist)

        # Count above unit circle
        n_above_unit = np.sum(magnitude_hist[magnitude_bins[:-1] > 1.0])

        extra_features = np.array(
            [
                angle_mean,
                angle_std,
                angle_peak,
                mag_mean,
                mag_std,
                mag_peak,
                n_above_unit,
            ]
        )

        signature = np.concatenate([signature, extra_features])

    return signature


def extract_signatures(
    models: List[Dict], signature_type: str = "histogram"
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Extract feature signatures from models.

    Args:
        models: List of model dicts from load_eigenvalues_from_summary
        signature_type: Type of signature ('histogram', 'statistics', 'combined')

    Returns:
        signatures: Array of signatures (n_models, n_features)
        model_names: List of model names
        metrics: Dict of metric arrays
    """
    signatures = []
    model_names = []
    metrics = {"accuracy": [], "frob_error": [], "spectral_error": []}

    for model in models:
        # Extract signature based on type
        if signature_type == "histogram":
            if "angle_histogram" not in model or "magnitude_histogram" not in model:
                continue
            sig = np.concatenate(
                [model["angle_histogram"], model["magnitude_histogram"]]
            )

        elif signature_type == "statistics":
            if "eigenvalues" not in model:
                continue
            stats = compute_eigenvalue_statistics(model["eigenvalues"])
            sig = np.array(list(stats.values()))

        elif signature_type == "combined":
            if "angle_histogram" not in model or "magnitude_histogram" not in model:
                continue
            sig = create_signature_from_histograms(
                model["angle_histogram"],
                model["magnitude_histogram"],
                model.get("angle_bins"),
                model.get("magnitude_bins"),
                include_statistics=True,
            )

        else:
            raise ValueError(f"Unknown signature type: {signature_type}")

        signatures.append(sig)
        model_names.append(model["model_name"])
        metrics["accuracy"].append(model["accuracy"])
        metrics["frob_error"].append(model["frob_error"])
        metrics["spectral_error"].append(model["spectral_error"])

    return np.array(signatures), model_names, metrics


def compare_eigenvalue_distributions(
    evals1: np.ndarray, evals2: np.ndarray, method: str = "wasserstein"
) -> float:
    """
    Compare two eigenvalue distributions.

    Args:
        evals1, evals2: Arrays of complex eigenvalues
        method: Comparison method ('wasserstein', 'hausdorff', 'emd')

    Returns:
        Distance between distributions
    """
    if method == "wasserstein":
        # Compute Wasserstein distance in magnitude and angle separately
        mag1, mag2 = np.abs(evals1), np.abs(evals2)
        angle1, angle2 = np.angle(evals1), np.angle(evals2)

        dist_mag = wasserstein_distance(mag1, mag2)
        dist_angle = wasserstein_distance(angle1, angle2)

        return dist_mag + dist_angle

    elif method == "hausdorff":
        from scipy.spatial.distance import directed_hausdorff

        points1 = np.column_stack([evals1.real, evals1.imag])
        points2 = np.column_stack([evals2.real, evals2.imag])

        return max(
            directed_hausdorff(points1, points2)[0],
            directed_hausdorff(points2, points1)[0],
        )

    else:
        raise ValueError(f"Unknown comparison method: {method}")
