"""
Procrustes Analysis for RNN Eigenvalue Distributions

Compares eigenvalue distributions using rotation-invariant Procrustes distance
and investigates correlation with Phi matrix similarity.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import procrustes
from sklearn.manifold import MDS
from tqdm import tqdm


def load_data(summary_file, seq_length=8, load_full_eigenvalues=True):
    """Load eigenvalues and Phi matrices from evaluation summary."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))

    with open(summary_file) as f:
        summary = json.load(f)

    models_dir = Path(summary_file).parent

    models = []
    for model_data in summary["metrics"]:
        model_name = model_data["model_name"]
        hidden_dim = model_data.get("hidden_dim", 0)

        # Try to load full W_hh eigenvalues from model checkpoint
        eigenvalues = None
        if load_full_eigenvalues:
            try:
                # Import pytorch and model code
                import torch

                from em_discrete.models.rnn_model import RNNModel

                # Find checkpoint (model_path is relative to project root)
                model_path = model_data.get("model_path", "")
                if model_path:
                    checkpoint_path = Path(model_path)

                    # Load model
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    state_dict = checkpoint.get("state_dict", checkpoint)

                    # Extract W_hh (try both with and without 'model.' prefix)
                    w_hh_key = None
                    if "model.rnn.weight_hh_l0" in state_dict:
                        w_hh_key = "model.rnn.weight_hh_l0"
                    elif "rnn.weight_hh_l0" in state_dict:
                        w_hh_key = "rnn.weight_hh_l0"

                    if w_hh_key:
                        W_hh = state_dict[w_hh_key].numpy()
                        # Compute full eigenvalues
                        eigenvalues = np.linalg.eigvals(W_hh)
                    else:
                        print(f"Warning: Could not find W_hh in {model_name}")
            except Exception as e:
                print(f"Warning: Could not load full eigenvalues for {model_name}: {e}")

        # Fallback to summary eigenvalues if full loading failed
        if eigenvalues is None:
            seq_key = f"seq{seq_length}"
            eigenvalue_key = f"eigenvalues_rnn_{seq_key}"
            if eigenvalue_key in model_data:
                evals_list = model_data[eigenvalue_key]
                eigenvalues = np.array([complex(e[0], e[1]) for e in evals_list])
            else:
                continue

        models.append(
            {
                "name": model_name,
                "eigenvalues": eigenvalues,
                "phi_matrix": None,  # Not used for now
                "accuracy": model_data.get("accuracy", 0.0),
                "hidden_dim": hidden_dim,
            }
        )

    return models


def eigenvalues_to_points(eigenvalues):
    """Convert complex eigenvalues to 2D points (real, imag) sorted by magnitude."""
    # Sort by magnitude for consistent correspondence
    sorted_evals = eigenvalues[np.argsort(np.abs(eigenvalues))[::-1]]
    return np.column_stack([sorted_evals.real, sorted_evals.imag])


def compute_procrustes_distance(points1, points2):
    """
    Compute Procrustes distance between two point sets.
    Returns disparity (standardized Procrustes distance).
    """
    _, _, disparity = procrustes(points1, points2)
    return disparity


def compute_phi_distance(phi1, phi2):
    """Compute Frobenius norm distance between two Phi matrices."""
    if phi1 is None or phi2 is None:
        return np.nan
    return np.linalg.norm(phi1 - phi2, "fro")


def compute_distance_matrices(models):
    """Compute Procrustes and Phi distance matrices."""
    n = len(models)

    # Convert eigenvalues to point clouds
    point_clouds = [eigenvalues_to_points(m["eigenvalues"]) for m in models]

    # Compute Procrustes distances
    print("Computing Procrustes distances...")
    procrustes_dist = np.zeros((n, n))
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            dist = compute_procrustes_distance(point_clouds[i], point_clouds[j])
            procrustes_dist[i, j] = dist
            procrustes_dist[j, i] = dist

    # Compute Phi distances
    print("Computing Phi matrix distances...")
    phi_dist = np.zeros((n, n))
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            dist = compute_phi_distance(
                models[i]["phi_matrix"], models[j]["phi_matrix"]
            )
            phi_dist[i, j] = dist
            phi_dist[j, i] = dist

    # Ensure symmetry (fix numerical precision issues)
    procrustes_dist = (procrustes_dist + procrustes_dist.T) / 2
    phi_dist = (phi_dist + phi_dist.T) / 2

    return procrustes_dist, phi_dist


def create_embedding_plot(distance_matrix, models, title, filename):
    """Create MDS embedding plot colored by accuracy."""
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=10)
    embedding = mds.fit_transform(distance_matrix)

    accuracies = np.array([m["accuracy"] for m in models])

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=accuracies,
        cmap="viridis",
        s=100,
        alpha=0.7,
        edgecolors="k",
        linewidth=0.5,
    )

    ax.set_xlabel("MDS Dimension 1", fontsize=11)
    ax.set_ylabel("MDS Dimension 2", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Accuracy", fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def create_correlation_plot(procrustes_dist, phi_dist, filename):
    """Plot correlation between Procrustes and Phi distances."""
    # Flatten upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(procrustes_dist, dtype=bool), k=1)
    proc_flat = procrustes_dist[mask]
    phi_flat = phi_dist[mask]

    # Remove NaN values
    valid = ~np.isnan(phi_flat)
    proc_flat = proc_flat[valid]
    phi_flat = phi_flat[valid]

    if len(proc_flat) == 0:
        print("Warning: No valid Phi distances for correlation analysis")
        return

    # Compute correlation
    corr = np.corrcoef(proc_flat, phi_flat)[0, 1]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Hexbin for density
    hb = ax.hexbin(proc_flat, phi_flat, gridsize=30, cmap="Blues", mincnt=1)

    # Fit line
    z = np.polyfit(proc_flat, phi_flat, 1)
    p = np.poly1d(z)
    x_line = np.linspace(proc_flat.min(), proc_flat.max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, label="Linear fit")

    ax.set_xlabel("Procrustes Distance (Eigenvalues)", fontsize=11)
    ax.set_ylabel("Frobenius Distance (Φ Matrices)", fontsize=11)
    ax.set_title(f"Correlation: r = {corr:.3f}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label("Count", fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    return corr


def create_comparison_plot(procrustes_dist, phi_dist, models, filename):
    """Create multi-panel comparison plot."""
    fig = plt.figure(figsize=(16, 5))

    # Panel 1: Procrustes MDS
    ax1 = plt.subplot(131)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=10)
    emb_proc = mds.fit_transform(procrustes_dist)
    accuracies = np.array([m["accuracy"] for m in models])
    scatter1 = ax1.scatter(
        emb_proc[:, 0],
        emb_proc[:, 1],
        c=accuracies,
        cmap="viridis",
        s=80,
        alpha=0.7,
        edgecolors="k",
        linewidth=0.5,
    )
    ax1.set_xlabel("MDS Dim 1", fontsize=10)
    ax1.set_ylabel("MDS Dim 2", fontsize=10)
    ax1.set_title("Procrustes Distance\n(Eigenvalues)", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Phi MDS (if available)
    ax2 = plt.subplot(132)
    if not np.all(np.isnan(phi_dist)):
        # Replace NaN with large value for MDS
        phi_dist_clean = phi_dist.copy()
        phi_dist_clean[np.isnan(phi_dist_clean)] = np.nanmax(phi_dist) * 2

        mds2 = MDS(
            n_components=2, dissimilarity="precomputed", random_state=42, n_init=10
        )
        emb_phi = mds2.fit_transform(phi_dist_clean)
        scatter2 = ax2.scatter(
            emb_phi[:, 0],
            emb_phi[:, 1],
            c=accuracies,
            cmap="viridis",
            s=80,
            alpha=0.7,
            edgecolors="k",
            linewidth=0.5,
        )
        ax2.set_xlabel("MDS Dim 1", fontsize=10)
        ax2.set_ylabel("MDS Dim 2", fontsize=10)
        ax2.set_title(
            "Frobenius Distance\n(Φ Matrices)", fontsize=11, fontweight="bold"
        )
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No Φ data available", ha="center", va="center", fontsize=12)
        ax2.set_xticks([])
        ax2.set_yticks([])

    # Panel 3: Correlation
    ax3 = plt.subplot(133)
    mask = np.triu(np.ones_like(procrustes_dist, dtype=bool), k=1)
    proc_flat = procrustes_dist[mask]
    phi_flat = phi_dist[mask]
    valid = ~np.isnan(phi_flat)

    if valid.sum() > 0:
        proc_flat = proc_flat[valid]
        phi_flat = phi_flat[valid]
        corr = np.corrcoef(proc_flat, phi_flat)[0, 1]

        ax3.hexbin(proc_flat, phi_flat, gridsize=25, cmap="Blues", mincnt=1)
        z = np.polyfit(proc_flat, phi_flat, 1)
        p = np.poly1d(z)
        x_line = np.linspace(proc_flat.min(), proc_flat.max(), 100)
        ax3.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8)
        ax3.set_xlabel("Procrustes Dist", fontsize=10)
        ax3.set_ylabel("Frobenius Dist", fontsize=10)
        ax3.set_title(f"Correlation\nr = {corr:.3f}", fontsize=11, fontweight="bold")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No correlation data", ha="center", va="center", fontsize=12)
        ax3.set_xticks([])
        ax3.set_yticks([])

    # Shared colorbar
    fig.colorbar(
        scatter1, ax=[ax1, ax2, ax3], label="Accuracy", fraction=0.02, pad=0.04
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def create_distance_heatmap(distance_matrix, models, title, filename):
    """Create clustered heatmap of distance matrix."""
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    # Ensure perfect symmetry and handle NaN
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)

    # Check if matrix has too many NaN values
    if np.isnan(distance_matrix).sum() > len(distance_matrix) * 2:
        print(f"Warning: Skipping heatmap for {title} due to too many NaN values")
        return

    # Perform hierarchical clustering
    condensed = squareform(distance_matrix)
    linkage_matrix = linkage(condensed, method="ward")

    fig = plt.figure(figsize=(12, 10))

    # Dendrogram
    ax1 = plt.subplot(221)
    dend = dendrogram(linkage_matrix, no_labels=True, ax=ax1)
    ax1.set_title("Hierarchical Clustering", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Model Index", fontsize=10)
    ax1.set_ylabel("Distance", fontsize=10)

    # Reorder distance matrix by clustering
    order = dend["leaves"]
    dist_ordered = distance_matrix[order, :][:, order]

    # Heatmap
    ax2 = plt.subplot(222)
    im = ax2.imshow(dist_ordered, cmap="viridis", aspect="auto")
    ax2.set_title(title, fontsize=11, fontweight="bold")
    ax2.set_xlabel("Model (ordered)", fontsize=10)
    ax2.set_ylabel("Model (ordered)", fontsize=10)
    plt.colorbar(im, ax=ax2, label="Distance")

    # Distance distribution
    ax3 = plt.subplot(223)
    mask = np.triu(np.ones_like(distance_matrix, dtype=bool), k=1)
    dists = distance_matrix[mask]
    dists = dists[~np.isnan(dists)]
    ax3.hist(dists, bins=40, alpha=0.7, edgecolor="black")
    ax3.axvline(
        np.median(dists),
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(dists):.3f}",
    )
    ax3.set_xlabel("Distance", fontsize=10)
    ax3.set_ylabel("Count", fontsize=10)
    ax3.set_title("Distance Distribution", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Accuracy vs cluster
    ax4 = plt.subplot(224)
    accuracies = np.array([models[i]["accuracy"] for i in order])
    ax4.plot(accuracies, "o-", alpha=0.7, markersize=4)
    ax4.set_xlabel("Model (ordered by clustering)", fontsize=10)
    ax4.set_ylabel("Accuracy", fontsize=10)
    ax4.set_title("Accuracy vs Cluster Order", fontsize=11, fontweight="bold")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Procrustes analysis of RNN eigenvalues"
    )
    parser.add_argument(
        "--summary_file", required=True, help="Path to evaluation_summary.json"
    )
    parser.add_argument(
        "--output_dir", default="procrustes_analysis", help="Output directory"
    )
    parser.add_argument("--seq_length", type=int, default=8, help="Sequence length")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Loading data from {args.summary_file}")
    models = load_data(args.summary_file, args.seq_length)
    print(f"Loaded {len(models)} models")
    print(
        f"Hidden dim: {models[0]['hidden_dim']}, Eigenvalues: {len(models[0]['eigenvalues'])}"
    )

    # Compute distance matrices
    procrustes_dist, phi_dist = compute_distance_matrices(models)

    # Save distance matrices
    np.save(output_dir / "procrustes_distances.npy", procrustes_dist)
    np.save(output_dir / "phi_distances.npy", phi_dist)
    print(f"\nSaved distance matrices to {output_dir}")

    # Create visualizations
    print("\nGenerating plots...")

    create_comparison_plot(
        procrustes_dist, phi_dist, models, output_dir / "procrustes_comparison.png"
    )
    print(f"  ✓ Comparison plot")

    create_distance_heatmap(
        procrustes_dist,
        models,
        "Procrustes Distance Matrix",
        output_dir / "procrustes_heatmap.png",
    )
    print(f"  ✓ Procrustes heatmap")

    if not np.all(np.isnan(phi_dist)):
        create_distance_heatmap(
            phi_dist, models, "Φ Matrix Distance Matrix", output_dir / "phi_heatmap.png"
        )
        print(f"  ✓ Phi heatmap")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    mask = np.triu(np.ones_like(procrustes_dist, dtype=bool), k=1)
    proc_vals = procrustes_dist[mask]
    print(f"\nProcrustes Distance:")
    print(f"  Mean:   {np.mean(proc_vals):.4f}")
    print(f"  Median: {np.median(proc_vals):.4f}")
    print(f"  Std:    {np.std(proc_vals):.4f}")
    print(f"  Range:  [{np.min(proc_vals):.4f}, {np.max(proc_vals):.4f}]")

    if not np.all(np.isnan(phi_dist)):
        phi_vals = phi_dist[mask]
        phi_vals = phi_vals[~np.isnan(phi_vals)]
        print(f"\nΦ Matrix Distance:")
        print(f"  Mean:   {np.mean(phi_vals):.4f}")
        print(f"  Median: {np.median(phi_vals):.4f}")
        print(f"  Std:    {np.std(phi_vals):.4f}")
        print(f"  Range:  [{np.min(phi_vals):.4f}, {np.max(phi_vals):.4f}]")

        # Correlation
        proc_flat = procrustes_dist[mask]
        phi_flat = phi_dist[mask]
        valid = ~np.isnan(phi_flat)
        if valid.sum() > 0:
            corr = np.corrcoef(proc_flat[valid], phi_flat[valid])[0, 1]
            print(f"\nCorrelation (Procrustes vs Φ): r = {corr:.4f}")

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_dir.absolute()}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
