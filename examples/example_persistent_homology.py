#!/usr/bin/env python3
"""
Example: Persistent Homology Analysis of Eigenvalue Distributions

This example demonstrates how to use persistent homology to analyze
the topological structure of eigenvalue distributions in the complex plane.

Usage:
    python examples/example_persistent_homology.py --summary_file path/to/evaluation_summary.json

Requirements:
    pip install ripser persim scikit-tda
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import analysis modules
from analysis import eigenvalue_analysis, persistent_homology, visualization


def parse_args():
    parser = argparse.ArgumentParser(description="Persistent homology example")
    parser.add_argument(
        "--summary_file",
        type=str,
        default="../50_models_seqlen_8_hidden_64/evaluation_summary.json",
        help="Path to evaluation_summary.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="persistence_example_output",
        help="Output directory",
    )
    parser.add_argument("--seq_length", type=int, default=8, help="Sequence length")
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("PERSISTENT HOMOLOGY EXAMPLE")
    print("=" * 70)
    print()

    # Check if ripser is available
    if not persistent_homology.RIPSER_AVAILABLE:
        print("ERROR: This example requires ripser and persim")
        print("Install with: pip install ripser persim scikit-tda")
        return

    # 1. Load eigenvalues
    print("1. Loading eigenvalues...")
    models = eigenvalue_analysis.load_eigenvalues_from_summary(
        args.summary_file, args.seq_length
    )
    print(f"   Loaded {len(models)} models")

    # Filter models with eigenvalues
    models_with_evals = [m for m in models if "eigenvalues" in m]
    print(f"   {len(models_with_evals)} have eigenvalue data")

    # 2. Visualize eigenvalues in complex plane (first 3 models)
    print("\n2. Visualizing eigenvalues in complex plane...")
    sample_models = models_with_evals[:3]
    evals_list = [m["eigenvalues"] for m in sample_models]
    labels = [m["model_name"] for m in sample_models]

    visualization.plot_eigenvalues_complex_plane(
        evals_list,
        labels=labels,
        title="Example Eigenvalue Distributions",
        output_path=output_dir / "eigenvalues_complex_plane.png",
    )
    print(f"   Saved to: {output_dir}/eigenvalues_complex_plane.png")

    # 3. Compute persistence diagrams
    print("\n3. Computing persistence diagrams...")
    diagrams = []
    for i, model in enumerate(models_with_evals[:5]):  # First 5 for example
        print(f"   Computing for {model['model_name']}...")
        dgm = persistent_homology.compute_persistence_diagram(
            model["eigenvalues"], max_dimension=1
        )
        diagrams.append(dgm)

    # 4. Plot example persistence diagrams
    print("\n4. Plotting persistence diagrams...")
    persistent_homology.plot_example_persistence_diagrams(
        models_with_evals[:5],
        diagrams,
        output_dir / "persistence_diagrams_examples.png",
        n_examples=5,
    )

    # 5. Compute topological statistics
    print("\n5. Computing topological statistics...")
    all_stats = []
    for model in models_with_evals:
        dgm = persistent_homology.compute_persistence_diagram(
            model["eigenvalues"], max_dimension=1
        )
        stats = persistent_homology.persistence_statistics(dgm)
        all_stats.append(stats)

    # Print summary statistics
    print("\n   Summary statistics across all models:")
    h0_counts = [s["h0_count"] for s in all_stats]
    h1_counts = [s["h1_count"] for s in all_stats]
    h0_max_pers = [s["h0_max_persistence"] for s in all_stats]
    h1_max_pers = [s["h1_max_persistence"] for s in all_stats]

    print(f"   H0 (connected components):")
    print(f"     Mean count: {np.mean(h0_counts):.2f} ± {np.std(h0_counts):.2f}")
    print(
        f"     Mean max persistence: {np.mean(h0_max_pers):.4f} ± {np.std(h0_max_pers):.4f}"
    )
    print(f"   H1 (loops):")
    print(f"     Mean count: {np.mean(h1_counts):.2f} ± {np.std(h1_counts):.2f}")
    print(
        f"     Mean max persistence: {np.mean(h1_max_pers):.4f} ± {np.std(h1_max_pers):.4f}"
    )

    # 6. Plot topological features
    print("\n6. Plotting topological feature distributions...")
    persistent_homology.plot_topological_features(
        models_with_evals,
        all_stats,
        output_dir / "topological_features.png",
    )

    # 7. Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("What do these features mean?")
    print()
    print("H0 (Connected Components):")
    print("  - Each eigenvalue starts as its own component (birth at scale 0)")
    print("  - Components merge as the scale increases")
    print("  - High persistence = well-separated clusters")
    print("  - Low persistence = tightly packed eigenvalues")
    print()
    print("H1 (Loops):")
    print("  - Circular arrangements of eigenvalues create loops")
    print("  - Eigenvalues on unit circle form prominent loops")
    print("  - Persistence = 'thickness' of the loop")
    print("  - Multiple loops suggest complex structure")
    print()
    print("If all models have similar topological features:")
    print("  → Eigenvalue distributions are structurally similar")
    print()
    print("If models cluster by topological features:")
    print("  → Different learning dynamics or solutions")
    print()

    # 8. Advanced: Compute pairwise bottleneck distances
    print("8. Computing pairwise bottleneck distances (this may take a while)...")
    print("   Computing for first 10 models as example...")

    sample_diagrams = []
    sample_models = models_with_evals[:10]

    for model in sample_models:
        dgm = persistent_homology.compute_persistence_diagram(
            model["eigenvalues"], max_dimension=1
        )
        sample_diagrams.append(dgm)

    # Compute distance matrix
    n = len(sample_diagrams)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = persistent_homology.compute_diagram_distance(
                sample_diagrams[i], sample_diagrams[j], metric="bottleneck"
            )
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    print(f"   Distance matrix shape: {dist_matrix.shape}")
    print(f"   Mean distance: {np.mean(dist_matrix[np.triu_indices(n, k=1)]):.4f}")

    # Plot distance matrix
    from analysis.visualization import plot_distance_matrix

    labels = [m["model_name"] for m in sample_models]
    plot_distance_matrix(
        dist_matrix,
        labels=labels,
        title="Bottleneck Distance Matrix (First 10 Models)",
        output_path=output_dir / "bottleneck_distance_matrix.png",
    )

    print()
    print("=" * 70)
    print("EXAMPLE COMPLETE")
    print("=" * 70)
    print(f"\nAll results saved to: {output_dir}/")
    print()
    print("Key outputs:")
    print(f"  - eigenvalues_complex_plane.png: Eigenvalues plotted in ℂ")
    print(f"  - persistence_diagrams_examples.png: Sample persistence diagrams")
    print(f"  - topological_features.png: Feature distributions")
    print(f"  - bottleneck_distance_matrix.png: Pairwise distances")
    print()
    print("Next steps:")
    print("  - Run full analysis: python analyze.py persistence --summary_file ...")
    print("  - Try different max_dimension values (0, 1, 2)")
    print("  - Compare multiple experiments using persistence distances")
    print()


if __name__ == "__main__":
    main()
