from pathlib import Path

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

from em_discrete.tasks.variable_linear_vb import BinaryLinearVarVBTask
from em_discrete.models.rnn_model import RNNModel
from emt_tools.models.linearModel import LinearModel


def identify_unique_fixed_points(fixed_points, residuals, threshold=0.01):
    """
    Identify unique fixed points by clustering similar ones.

    Args:
        fixed_points: tensor of shape (hidden_dim, num_fps)
        residuals: tensor of shape (num_fps,)
        threshold: distance threshold for considering points as the same

    Returns:
        unique_fps: unique fixed points
        unique_indices: indices of unique fixed points
        labels: cluster labels for each fixed point
    """
    # Filter out poorly converged fixed points
    good_mask = residuals < 0.1
    good_fps = fixed_points[:, good_mask]
    good_indices = torch.where(good_mask)[0]

    if good_fps.shape[1] == 0:
        return torch.empty(fixed_points.shape[0], 0), torch.tensor([]), torch.tensor([])

    # Compute pairwise distances
    fps_np = good_fps.T.numpy()  # (num_fps, hidden_dim)

    # Use DBSCAN to cluster similar fixed points
    clustering = DBSCAN(eps=threshold, min_samples=1, metric='euclidean')
    labels = clustering.fit_predict(fps_np)

    # Select one representative from each cluster
    unique_indices = []
    for label in np.unique(labels):
        cluster_mask = labels == label
        cluster_indices = good_indices[cluster_mask]
        # Pick the one with smallest residual
        best_idx = cluster_indices[residuals[cluster_indices].argmin()]
        unique_indices.append(best_idx.item())

    unique_indices = torch.tensor(unique_indices)
    unique_fps = fixed_points[:, unique_indices]

    return unique_fps, unique_indices, torch.tensor(labels)


def analyze_fixed_point_stability(W_hh, fixed_points):
    """
    Analyze stability of fixed points by computing eigenvalues of the Jacobian.

    For h* = tanh(W_hh @ h*), the Jacobian is:
    J = diag(1 - tanh(W_hh @ h*)^2) @ W_hh

    A fixed point is stable if all eigenvalues have magnitude < 1.
    """
    hidden_dim = fixed_points.shape[0]
    num_fps = fixed_points.shape[1]

    stability_info = []

    for i in range(num_fps):
        h_star = fixed_points[:, i]

        # Compute Jacobian: J = diag(sech^2(pre_activation)) @ W_hh
        pre_activation = W_hh @ h_star
        activation = torch.tanh(pre_activation)
        sech_squared = 1 - activation**2  # derivative of tanh

        # Jacobian matrix
        J = torch.diag(sech_squared) @ W_hh

        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvals(J)
        max_eigenvalue = torch.max(torch.abs(eigenvalues))

        is_stable = max_eigenvalue < 1.0

        stability_info.append({
            'index': i,
            'max_eigenvalue': max_eigenvalue.item(),
            'is_stable': is_stable.item(),
            'eigenvalues': eigenvalues.numpy()
        })

    return stability_info


def visualize_fixed_points(fixed_points, residuals, unique_indices, stability_info,
                           W_hh, output_dir):
    """Create comprehensive visualizations of fixed points."""
    output_dir = Path(output_dir)

    num_fps = fixed_points.shape[1]
    fps_np = fixed_points.T.numpy()  # (num_fps, hidden_dim)

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))

    # 1. Distance matrix
    ax1 = plt.subplot(2, 4, 1)
    distances = squareform(pdist(fps_np, metric='euclidean'))
    im1 = ax1.imshow(distances, cmap='viridis')
    ax1.set_title('Pairwise Distance Matrix')
    ax1.set_xlabel('Fixed Point Index')
    ax1.set_ylabel('Fixed Point Index')
    plt.colorbar(im1, ax=ax1)

    # 2. Hierarchical clustering dendrogram
    ax2 = plt.subplot(2, 4, 2)
    if num_fps > 1:
        linkage_matrix = linkage(fps_np, method='ward')
        dendrogram(linkage_matrix, ax=ax2)
        ax2.set_title('Hierarchical Clustering')
        ax2.set_xlabel('Fixed Point Index')
        ax2.set_ylabel('Distance')
    else:
        ax2.text(0.5, 0.5, 'Need >1 FP', ha='center', va='center')
        ax2.set_title('Hierarchical Clustering')

    # 3. PCA projection (2D)
    ax3 = plt.subplot(2, 4, 3)
    if num_fps > 1 and fps_np.shape[1] > 2:
        pca = PCA(n_components=2)
        fps_pca = pca.fit_transform(fps_np)

        colors = ['red' if i in unique_indices else 'blue' for i in range(num_fps)]
        sizes = [100 if i in unique_indices else 30 for i in range(num_fps)]

        ax3.scatter(fps_pca[:, 0], fps_pca[:, 1], c=colors, s=sizes, alpha=0.6)

        # Label unique fixed points
        for idx in unique_indices:
            ax3.annotate(f'FP{idx}', (fps_pca[idx, 0], fps_pca[idx, 1]),
                        fontsize=8, fontweight='bold')

        ax3.set_title(f'PCA Projection (Var: {pca.explained_variance_ratio_.sum():.2%})')
        ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Insufficient dims', ha='center', va='center')
        ax3.set_title('PCA Projection')

    # 4. Residuals
    ax4 = plt.subplot(2, 4, 4)
    colors = ['red' if i in unique_indices else 'blue' for i in range(num_fps)]
    ax4.bar(range(num_fps), residuals.numpy(), color=colors, alpha=0.6)
    ax4.axhline(y=0.01, color='orange', linestyle='--', label='Good convergence')
    ax4.set_title('Fixed Point Residuals')
    ax4.set_xlabel('Fixed Point Index')
    ax4.set_ylabel('Residual')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Fixed point heatmap
    ax5 = plt.subplot(2, 4, 5)
    im5 = ax5.imshow(fixed_points.numpy(), aspect='auto', cmap='RdBu_r')
    ax5.set_title('All Fixed Points (Hidden State)')
    ax5.set_xlabel('Fixed Point Index')
    ax5.set_ylabel('Hidden Dimension')
    plt.colorbar(im5, ax=ax5)

    # 6. Unique fixed points heatmap
    ax6 = plt.subplot(2, 4, 6)
    if len(unique_indices) > 0:
        unique_fps = fixed_points[:, unique_indices].numpy()
        im6 = ax6.imshow(unique_fps, aspect='auto', cmap='RdBu_r')
        ax6.set_title(f'Unique Fixed Points ({len(unique_indices)})')
        ax6.set_xlabel('Unique FP Index')
        ax6.set_ylabel('Hidden Dimension')
        plt.colorbar(im6, ax=ax6)
    else:
        ax6.text(0.5, 0.5, 'No unique FPs', ha='center', va='center')
        ax6.set_title('Unique Fixed Points')

    # 7. Eigenvalue spectrum of W_hh
    ax7 = plt.subplot(2, 4, 7)
    eigenvalues_W = torch.linalg.eigvals(W_hh).numpy()
    ax7.scatter(eigenvalues_W.real, eigenvalues_W.imag, alpha=0.6)
    circle = plt.Circle((0, 0), 1.0, fill=False, color='red', linestyle='--',
                       label='Unit circle')
    ax7.add_patch(circle)
    ax7.set_title('Eigenvalues of W_hh')
    ax7.set_xlabel('Real')
    ax7.set_ylabel('Imaginary')
    ax7.axis('equal')
    ax7.grid(True, alpha=0.3)
    ax7.legend()

    # 8. Stability analysis
    ax8 = plt.subplot(2, 4, 8)
    if stability_info:
        max_eigs = [info['max_eigenvalue'] for info in stability_info]
        colors = ['green' if info['is_stable'] else 'red'
                 for info in stability_info]
        ax8.bar(range(len(stability_info)), max_eigs, color=colors, alpha=0.6)
        ax8.axhline(y=1.0, color='black', linestyle='--', label='Stability threshold')
        ax8.set_title('Stability Analysis')
        ax8.set_xlabel('Fixed Point Index')
        ax8.set_ylabel('Max |Eigenvalue| of Jacobian')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    else:
        ax8.text(0.5, 0.5, 'No stability info', ha='center', va='center')
        ax8.set_title('Stability Analysis')

    plt.tight_layout()
    fig.savefig(output_dir / "fixed_points_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved visualization to: {output_dir / 'fixed_points_analysis.png'}")


def load_model(model_path, device):
    """Load checkpoint and create model with correct architecture."""
    print(f"Loading checkpoint from: {model_path}")

    # Load checkpoint to inspect hyperparameters and state_dict
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    hparams = checkpoint["hyper_parameters"]
    state_dict = checkpoint.get("state_dict", {})

    # Extract basic parameters from hyperparameters
    data_dim = hparams.get("input_dim", 8)
    hidden_dim = hparams.get("hidden_dim", 128)
    batch_size = hparams.get("batch_size", 64)

    # Detect if model uses EOS token by inspecting actual weight shapes
    # This is more reliable than checking hyperparameters
    weight_ih_key = "model.rnn.weight_ih_l0"
    if weight_ih_key in state_dict:
        actual_input_dim = state_dict[weight_ih_key].shape[1]
        actual_hidden_dim = state_dict[weight_ih_key].shape[0]
    else:
        # Fallback to checking hyperparameters
        use_eos = hparams.get("use_eos", False)
        infinite_horizon = hparams.get("infinite_horizon", False)
        actual_input_dim = data_dim + 1 if (use_eos or infinite_horizon) else data_dim
        actual_hidden_dim = hidden_dim

    # Determine output dimension (always data_dim, never includes EOS)
    model_output_dim = data_dim

    print(
        f"Model config: data_dim={data_dim}, hidden_dim={hidden_dim}, batch_size={batch_size}"
    )
    print(
        f"Detected dimensions: model_input_dim={actual_input_dim}, model_output_dim={model_output_dim}"
    )

    # Check if model uses EOS/infinite horizon
    uses_eos = actual_input_dim == data_dim + 1
    # Detect infinite horizon by presence of specific hyperparameters
    infinite_horizon = "output_horizon" in hparams and "total_length" in hparams

    if uses_eos and not infinite_horizon:
        print("Note: Model uses EOS token (regular mode)")
    elif infinite_horizon:
        print("Note: Model was trained with infinite horizon mode (includes EOS)")

    # Create the model with the correct architecture
    model = RNNModel(actual_input_dim, actual_hidden_dim, model_output_dim, bias=False)

    # Load the checkpoint with the correct task class
    if infinite_horizon:
        # Import infinite horizon task
        from em_discrete.tasks.infinite_horizon_vb import (
            BinaryLinearVBInfiniteHorizonTask,
        )

        lmodule = BinaryLinearVBInfiniteHorizonTask.load_from_checkpoint(
            model_path, model=model, map_location=torch.device(device)
        )
    else:
        lmodule = BinaryLinearVarVBTask.load_from_checkpoint(
            model_path, model=model, map_location=torch.device(device)
        )

    lmodule.eval()

    # Create EMT linear model for analysis
    emt_linear_model = LinearModel(lmodule.input_dim, lmodule.model.hidden_dim)
    emt_linear_model.parse_simple_rnn(lmodule.model)

    return lmodule, emt_linear_model, hparams, batch_size


@click.command()
@click.option(
    "--ckpt_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option("--num_fixed_points", default=10, help="Number of fixed points to find")
@click.option("--lr", default=1e-2, help="Learning rate")
@click.option("--n_steps", default=10_000, help="Number of optimization steps")
@click.option(
    "--input", default=None, type=float, multiple=True, help="Optional input vector"
)
@click.option("--cluster_threshold", default=0.5, help="Distance threshold for clustering unique FPs")
def main(ckpt_path: Path, num_fixed_points: int, lr: float, n_steps: int, input, cluster_threshold: float):
    lmodule, linear_model, hparams, batch_size = load_model(ckpt_path, "cpu")

    # Model weights (frozen - not optimized)
    W_ih = torch.tensor(linear_model.W_uh, dtype=torch.float32)  # input-to-hidden
    W_hh = torch.tensor(linear_model.W_hh, dtype=torch.float32)  # hidden-to-hidden
    W_hy = torch.tensor(linear_model.W_hy, dtype=torch.float32)  # hidden-to-output

    hidden_dim = W_hh.shape[0]

    # Handle input
    if input:
        u = (
            torch.tensor(input, dtype=torch.float32)
            .unsqueeze(1)
            .expand(-1, num_fixed_points)
        )
        print(f"Finding fixed points with input: {input}")
    else:
        u = torch.zeros(W_ih.shape[1], num_fixed_points)  # shape[1] is input_dim
        print("Finding fixed points with zero input")

    # Initialize hidden states to optimize (these are the parameters we optimize!)
    h = nn.Parameter(torch.randn(hidden_dim, num_fixed_points) * 0.1)

    optimizer = torch.optim.Adam([h], lr=lr)

    print(f"Starting optimization for {num_fixed_points} fixed points...")
    print(f"Hidden dim: {hidden_dim}, Steps: {n_steps}, LR: {lr}")

    for step in range(n_steps):
        optimizer.zero_grad()

        # RNN dynamics: h_next = tanh(W_ih @ u + W_hh @ h)
        h_next = torch.tanh(W_ih @ u + W_hh @ h)

        # Fixed point constraint: h should equal h_next
        loss = F.mse_loss(h, h_next)

        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == n_steps - 1:
            with torch.no_grad():
                # Compute residual norm for each fixed point
                residual = torch.norm(h - h_next, dim=0)
                print(
                    f"Step {step:5d} | Loss: {loss.item():.6f} | "
                    f"Max residual: {residual.max().item():.6f} | "
                    f"Min residual: {residual.min().item():.6f}"
                )

    # Final results
    print("\n" + "=" * 60)
    print("Fixed Point Finding Complete")
    print("=" * 60)

    with torch.no_grad():
        h_final = h.detach()
        h_next_final = torch.tanh(W_ih @ u + W_hh @ h_final)
        residuals = torch.norm(h_final - h_next_final, dim=0)

        print(f"\nFound {num_fixed_points} fixed points:")
        for i in range(num_fixed_points):
            print(f"  FP {i+1}: residual = {residuals[i].item():.8f}")

        # Identify unique fixed points
        print("\n" + "="*60)
        print("Identifying Unique Fixed Points")
        print("="*60)
        unique_fps, unique_indices, labels = identify_unique_fixed_points(
            h_final, residuals, threshold=cluster_threshold
        )
        print(f"\nClustering threshold: {cluster_threshold}")
        print(f"Number of distinct fixed points: {len(unique_indices)}")
        print(f"Indices of unique fixed points: {unique_indices.tolist()}")

        # Analyze stability
        print("\n" + "="*60)
        print("Stability Analysis")
        print("="*60)
        stability_info = analyze_fixed_point_stability(W_hh, h_final)

        num_stable = sum(1 for info in stability_info if info['is_stable'])
        print(f"\nStable fixed points: {num_stable}/{len(stability_info)}")

        for info in stability_info:
            stability_str = "STABLE" if info['is_stable'] else "UNSTABLE"
            print(f"  FP {info['index']:2d}: max|λ| = {info['max_eigenvalue']:.4f} [{stability_str}]")

        # Create visualizations
        print("\n" + "="*60)
        print("Creating Visualizations")
        print("="*60)
        visualize_fixed_points(h_final, residuals, unique_indices, stability_info,
                              W_hh, ckpt_path.parent)

        # Save results
        output_path = ckpt_path.parent / "fixed_points.pt"
        torch.save(
            {
                "fixed_points": h_final,
                "residuals": residuals,
                "unique_indices": unique_indices,
                "unique_fixed_points": unique_fps,
                "stability_info": stability_info,
                "input": u,
                "W_ih": W_ih,
                "W_hh": W_hh,
                "W_hy": W_hy,
            },
            output_path,
        )
        print(f"\nSaved fixed points to: {output_path}")


if __name__ == "__main__":
    main()
