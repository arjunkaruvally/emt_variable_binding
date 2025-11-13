"""
Dimensionality reduction methods for model analysis.

This module provides utilities for reducing high-dimensional signatures
to 2D/3D for visualization and analysis.
"""

from typing import Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, Isomap


def apply_pca(
    data: np.ndarray, n_components: int = 2, return_model: bool = False
) -> Tuple:
    """
    Apply PCA dimensionality reduction.

    Args:
        data: Input data (n_samples, n_features)
        n_components: Number of components
        return_model: If True, return fitted model

    Returns:
        transformed_data: Reduced data
        explained_variance: Array of explained variance ratios
        model: (optional) Fitted PCA model
    """
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data)

    result = (transformed, pca.explained_variance_ratio_)

    if return_model:
        result = result + (pca,)

    return result


def apply_tsne(
    data: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Apply t-SNE dimensionality reduction.

    Args:
        data: Input data (n_samples, n_features) or precomputed distances
        n_components: Number of components
        perplexity: t-SNE perplexity parameter
        random_state: Random seed
        metric: Distance metric ('euclidean' or 'precomputed')

    Returns:
        Transformed data
    """
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        metric=metric,
    )
    return tsne.fit_transform(data)


def apply_mds(
    data: np.ndarray,
    n_components: int = 2,
    dissimilarity: str = "euclidean",
    random_state: int = 42,
) -> np.ndarray:
    """
    Apply MDS (Multidimensional Scaling).

    Args:
        data: Input data or distance matrix
        n_components: Number of components
        dissimilarity: 'euclidean' or 'precomputed'
        random_state: Random seed

    Returns:
        Transformed data
    """
    mds = MDS(
        n_components=n_components,
        dissimilarity=dissimilarity,
        random_state=random_state,
    )
    return mds.fit_transform(data)


def apply_isomap(
    data: np.ndarray, n_components: int = 2, n_neighbors: int = 5
) -> np.ndarray:
    """
    Apply Isomap dimensionality reduction.

    Args:
        data: Input data (n_samples, n_features)
        n_components: Number of components
        n_neighbors: Number of neighbors for graph construction

    Returns:
        Transformed data
    """
    isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    return isomap.fit_transform(data)


def compute_dimensionality_reduction(
    data: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    is_distance_matrix: bool = False,
    **kwargs,
) -> Tuple:
    """
    Unified interface for dimensionality reduction.

    Args:
        data: Input data or distance matrix
        method: 'pca', 'tsne', 'mds', 'isomap'
        n_components: Number of output dimensions
        is_distance_matrix: If True, treat data as precomputed distances
        **kwargs: Additional method-specific arguments

    Returns:
        transformed_data: Reduced data
        extra_info: Method-specific information (e.g., explained variance for PCA)
    """
    if method == "pca":
        if is_distance_matrix:
            raise ValueError("PCA cannot use precomputed distance matrix")
        transformed, explained_var = apply_pca(data, n_components)
        return transformed, {"explained_variance": explained_var}

    elif method == "tsne":
        metric = "precomputed" if is_distance_matrix else "euclidean"
        transformed = apply_tsne(data, n_components, metric=metric, **kwargs)
        return transformed, {}

    elif method == "mds":
        dissimilarity = "precomputed" if is_distance_matrix else "euclidean"
        transformed = apply_mds(
            data, n_components, dissimilarity=dissimilarity, **kwargs
        )
        return transformed, {}

    elif method == "isomap":
        if is_distance_matrix:
            raise ValueError("Isomap cannot use precomputed distance matrix")
        transformed = apply_isomap(data, n_components, **kwargs)
        return transformed, {}

    else:
        raise ValueError(f"Unknown method: {method}")


def analyze_pca_components(
    pca_model, feature_names: Optional[list] = None, n_top: int = 10
) -> dict:
    """
    Analyze PCA components to understand feature importance.

    Args:
        pca_model: Fitted PCA model
        feature_names: Names of features
        n_top: Number of top features to report per component

    Returns:
        Dictionary with component analysis
    """
    n_components = pca_model.n_components_
    components = pca_model.components_

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(components.shape[1])]

    analysis = {
        "explained_variance": pca_model.explained_variance_ratio_.tolist(),
        "components": [],
    }

    for i in range(n_components):
        component = components[i]

        # Get top positive and negative features
        sorted_indices = np.argsort(np.abs(component))[::-1]
        top_indices = sorted_indices[:n_top]

        top_features = [
            {
                "feature": feature_names[idx],
                "weight": float(component[idx]),
                "abs_weight": float(np.abs(component[idx])),
            }
            for idx in top_indices
        ]

        analysis["components"].append(
            {
                "component_id": i,
                "explained_variance": float(pca_model.explained_variance_ratio_[i]),
                "top_features": top_features,
            }
        )

    return analysis


def compute_intrinsic_dimensionality(
    data: np.ndarray, variance_threshold: float = 0.95
) -> int:
    """
    Estimate intrinsic dimensionality using PCA.

    Args:
        data: Input data
        variance_threshold: Cumulative variance threshold

    Returns:
        Number of components needed to explain variance_threshold of variance
    """
    pca = PCA()
    pca.fit(data)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.searchsorted(cumulative_variance, variance_threshold) + 1

    return n_components


def validate_embedding_quality(
    original_distances: np.ndarray, embedded_data: np.ndarray
) -> dict:
    """
    Validate quality of dimensionality reduction by comparing distances.

    Args:
        original_distances: Original pairwise distance matrix
        embedded_data: Embedded data points

    Returns:
        Dictionary of quality metrics
    """
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import pearsonr, spearmanr

    # Compute distances in embedded space
    embedded_distances = squareform(pdist(embedded_data))

    # Extract upper triangles
    n = original_distances.shape[0]
    triu_indices = np.triu_indices(n, k=1)

    orig_dists = original_distances[triu_indices]
    embed_dists = embedded_distances[triu_indices]

    # Correlation metrics
    pearson_r, pearson_p = pearsonr(orig_dists, embed_dists)
    spearman_r, spearman_p = spearmanr(orig_dists, embed_dists)

    # Stress (normalized)
    stress = np.sqrt(np.sum((orig_dists - embed_dists) ** 2) / np.sum(orig_dists**2))

    return {
        "pearson_correlation": float(pearson_r),
        "pearson_pvalue": float(pearson_p),
        "spearman_correlation": float(spearman_r),
        "spearman_pvalue": float(spearman_p),
        "stress": float(stress),
    }
