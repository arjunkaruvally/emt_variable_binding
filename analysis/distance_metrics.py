"""
Distance metrics for comparing model representations.

This module provides various distance and similarity metrics for comparing
eigenvalue distributions, signatures, and model properties.
"""

from typing import List

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance
from tqdm import tqdm


def sliced_wasserstein_distance_2d(
    points1: np.ndarray, points2: np.ndarray, n_projections: int = 50
) -> float:
    """
    Compute sliced Wasserstein distance for 2D point clouds.

    Args:
        points1, points2: 2D point clouds (n_points, 2)
        n_projections: Number of random projections

    Returns:
        Sliced Wasserstein distance
    """
    np.random.seed(42)
    distances = []

    for _ in range(n_projections):
        # Random direction on unit circle
        theta = np.random.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(theta), np.sin(theta)])

        # Project points onto direction
        proj1 = points1 @ direction
        proj2 = points2 @ direction

        # Compute 1D Wasserstein distance
        distances.append(wasserstein_distance(proj1, proj2))

    return np.mean(distances)


def wasserstein_eigenvalue_distance(
    evals1: np.ndarray, evals2: np.ndarray, mode: str = "separate"
) -> float:
    """
    Compute Wasserstein distance between eigenvalue distributions.

    Args:
        evals1, evals2: Complex eigenvalue arrays
        mode: 'separate' (angle + magnitude), '2d' (sliced in complex plane),
              'magnitude_only', 'angle_only'

    Returns:
        Distance value
    """
    if mode == "separate":
        # Separate distance for magnitude and angle
        mag1, mag2 = np.abs(evals1), np.abs(evals2)
        angle1, angle2 = np.angle(evals1), np.angle(evals2)

        dist_mag = wasserstein_distance(mag1, mag2)
        dist_angle = wasserstein_distance(angle1, angle2)

        return dist_mag + dist_angle

    elif mode == "2d":
        # 2D Wasserstein via sliced distance
        points1 = np.column_stack([evals1.real, evals1.imag])
        points2 = np.column_stack([evals2.real, evals2.imag])

        return sliced_wasserstein_distance_2d(points1, points2)

    elif mode == "magnitude_only":
        mag1, mag2 = np.abs(evals1), np.abs(evals2)
        return wasserstein_distance(mag1, mag2)

    elif mode == "angle_only":
        angle1, angle2 = np.angle(evals1), np.angle(evals2)
        return wasserstein_distance(angle1, angle2)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_pairwise_distances(
    data: np.ndarray, metric: str = "euclidean"
) -> np.ndarray:
    """
    Compute pairwise distances between data points.

    Args:
        data: Array of data points (n_samples, n_features)
        metric: Distance metric (any scipy pdist metric)

    Returns:
        Distance matrix (n_samples, n_samples)
    """
    distances = pdist(data, metric=metric)
    return squareform(distances)


def compute_eigenvalue_distance_matrix(
    eigenvalue_list: List[np.ndarray],
    mode: str = "separate",
    show_progress: bool = True,
) -> np.ndarray:
    """
    Compute pairwise Wasserstein distances between eigenvalue distributions.

    Args:
        eigenvalue_list: List of eigenvalue arrays
        mode: Wasserstein mode
        show_progress: Show progress bar

    Returns:
        Symmetric distance matrix
    """
    n = len(eigenvalue_list)
    distance_matrix = np.zeros((n, n))

    iterator = tqdm(range(n)) if show_progress else range(n)

    for i in iterator:
        for j in range(i + 1, n):
            dist = wasserstein_eigenvalue_distance(
                eigenvalue_list[i], eigenvalue_list[j], mode=mode
            )
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix


def histogram_distance(
    hist1: np.ndarray, hist2: np.ndarray, method: str = "euclidean"
) -> float:
    """
    Compute distance between histograms.

    Args:
        hist1, hist2: Histogram arrays (must have same length)
        method: 'euclidean', 'manhattan', 'chi2', 'kl', 'bhattacharyya'

    Returns:
        Distance value
    """
    if method == "euclidean":
        return np.linalg.norm(hist1 - hist2)

    elif method == "manhattan":
        return np.sum(np.abs(hist1 - hist2))

    elif method == "chi2":
        # Chi-squared distance
        return np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-10))

    elif method == "kl":
        # Kullback-Leibler divergence (not symmetric)
        hist1_norm = hist1 / (np.sum(hist1) + 1e-10)
        hist2_norm = hist2 / (np.sum(hist2) + 1e-10)
        return np.sum(hist1_norm * np.log((hist1_norm + 1e-10) / (hist2_norm + 1e-10)))

    elif method == "bhattacharyya":
        # Bhattacharyya distance
        hist1_norm = hist1 / (np.sum(hist1) + 1e-10)
        hist2_norm = hist2 / (np.sum(hist2) + 1e-10)
        bc = np.sum(np.sqrt(hist1_norm * hist2_norm))
        return -np.log(bc + 1e-10)

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_distance_statistics(distance_matrix: np.ndarray) -> dict:
    """
    Compute statistics of a distance matrix.

    Args:
        distance_matrix: Symmetric distance matrix

    Returns:
        Dictionary of statistics
    """
    # Extract upper triangle (excluding diagonal)
    n = distance_matrix.shape[0]
    upper_tri_indices = np.triu_indices(n, k=1)
    distances = distance_matrix[upper_tri_indices]

    stats = {
        "mean": np.mean(distances),
        "median": np.median(distances),
        "std": np.std(distances),
        "min": np.min(distances),
        "max": np.max(distances),
        "q25": np.percentile(distances, 25),
        "q75": np.percentile(distances, 75),
        "range": np.max(distances) - np.min(distances),
        "relative_spread": np.std(distances) / (np.mean(distances) + 1e-10),
        "n_pairs": len(distances),
    }

    return stats


def identify_outliers_by_distance(
    distance_matrix: np.ndarray, threshold: float = 2.0
) -> np.ndarray:
    """
    Identify outliers based on mean distance to all other points.

    Args:
        distance_matrix: Symmetric distance matrix
        threshold: Number of standard deviations for outlier detection

    Returns:
        Boolean array indicating outliers
    """
    mean_distances = np.mean(distance_matrix, axis=1)
    mean_of_means = np.mean(mean_distances)
    std_of_means = np.std(mean_distances)

    outlier_threshold = mean_of_means + threshold * std_of_means
    return mean_distances > outlier_threshold


def compute_nearest_neighbors(distance_matrix: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Find k nearest neighbors for each point.

    Args:
        distance_matrix: Symmetric distance matrix
        k: Number of neighbors

    Returns:
        Array of neighbor indices (n_samples, k)
    """
    n = distance_matrix.shape[0]
    neighbors = np.zeros((n, k), dtype=int)

    for i in range(n):
        # Get distances to all other points
        dists = distance_matrix[i].copy()
        dists[i] = np.inf  # Exclude self

        # Find k nearest
        neighbors[i] = np.argsort(dists)[:k]

    return neighbors
