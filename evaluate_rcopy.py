import os
import os.path as osp

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from emt_tools.models.linearModel import LinearModel
from emt_tools.utils import spectral_comparison
from mpl_toolkits.axes_grid1 import make_axes_locatable

from em_discrete.models.rnn_model import RNNModel
from em_discrete.tasks.variable_linear_vb import BinaryLinearVarVBTask


def task_specific_subspace(linear_model, task_dim, phi_theoretical):
    """
    Use PCA/SVD to identify the dominant subspace of W_hh and project
    the theoretical matrix into that subspace for comparison.
    """
    W_hh = linear_model.W_hh

    print(f"\n=== PCA-based Subspace Analysis ===")
    print(f"Full W_hh shape: {W_hh.shape}")
    print(f"Task dimension: {task_dim}")

    # Perform SVD on W_hh to find principal components
    U, S, Vh = np.linalg.svd(W_hh, full_matrices=True)

    # Analyze singular values to see how much variance is in top task_dim dimensions
    total_variance = np.sum(S**2)
    top_k_variance = np.sum(S[:task_dim] ** 2)
    variance_ratio = top_k_variance / total_variance

    print(f"\nSingular value analysis:")
    print(
        f"Top {task_dim} singular values capture {100 * variance_ratio:.2f}% of variance"
    )
    print(f"Top 10 singular values: {S[:10]}")
    print(
        f"Singular values {task_dim - 5} to {task_dim + 5}: {S[max(0, task_dim - 5) : task_dim + 5]}"
    )

    # Extract the top task_dim principal components
    # These define the dominant subspace
    U_dominant = U[:, :task_dim]  # First task_dim left singular vectors

    # Project W_hh into the dominant subspace
    # W_hh_projected = U_dominant.T @ W_hh @ U_dominant
    # This gives us the task_dim x task_dim matrix in the principal subspace
    W_hh_subspace = U_dominant.T @ W_hh @ U_dominant

    print(f"\nProjected matrices:")
    print(f"W_hh_subspace shape: {W_hh_subspace.shape}")
    print(f"phi_theoretical shape: {phi_theoretical.shape}")

    # Compute eigenvalues for diagnostics
    evals_rnn = np.linalg.eigvals(W_hh_subspace)
    evals_theory = np.linalg.eigvals(phi_theoretical)

    print(f"\nEigenvalue diagnostics:")
    print(
        f"RNN (PCA subspace) - count: {len(evals_rnn)}, |λ|>0.01: {np.sum(np.abs(evals_rnn) > 0.01)}"
    )
    print(
        f"Theory eigenvalues - count: {len(evals_theory)}, |λ|>0.01: {np.sum(np.abs(evals_theory) > 0.01)}"
    )
    print(
        f"RNN eigenvalue range: [{np.abs(evals_rnn).min():.4f}, {np.abs(evals_rnn).max():.4f}]"
    )
    print(
        f"Theory eigenvalue range: [{np.abs(evals_theory).min():.4f}, {np.abs(evals_theory).max():.4f}]"
    )

    # Filter out very small eigenvalues before comparison to avoid numerical issues
    threshold = 1e-6
    evals_rnn_filtered = evals_rnn[np.abs(evals_rnn) > threshold]
    evals_theory_filtered = evals_theory[np.abs(evals_theory) > threshold]

    print(f"\nAfter filtering (|λ| > {threshold}):")
    print(f"RNN: {len(evals_rnn_filtered)} eigenvalues")
    print(f"Theory: {len(evals_theory_filtered)} eigenvalues")

    # Try the built-in spectral comparison
    try:
        spectral_error = spectral_comparison(W_hh_subspace, phi_theoretical)
        if not np.isnan(spectral_error):
            print(f"\nSpectral error (in degrees): {np.degrees(spectral_error):.4f}")
        else:
            print("\nSpectral comparison returned NaN (numerical issues)")
    except Exception as e:
        print(f"\nSpectral comparison failed: {e}")

    # Compute alternative metrics for comparison
    print("\n=== Alternative Comparison Metrics (PCA Subspace) ===")

    # 1. Frobenius norm error
    frob_error = np.linalg.norm(W_hh_subspace - phi_theoretical, "fro")
    frob_relative = frob_error / np.linalg.norm(phi_theoretical, "fro")
    print(f"Frobenius norm error: {frob_error:.4f}")
    print(f"Relative Frobenius error: {frob_relative:.4f} ({100 * frob_relative:.2f}%)")

    # 2. Eigenvalue magnitude error
    evals_rnn_sorted = np.sort(np.abs(evals_rnn))[::-1]
    evals_theory_sorted = np.sort(np.abs(evals_theory))[::-1]
    eigenvalue_error = np.mean(np.abs(evals_rnn_sorted - evals_theory_sorted))
    print(f"Mean eigenvalue magnitude error: {eigenvalue_error:.4f}")

    # 3. Custom spectral angle (only for non-zero eigenvalues)
    def compute_spectral_angle(evals1, evals2, threshold=1e-6):
        """Compute average angular distance between eigenvalue sets."""
        # Filter out near-zero eigenvalues
        evals1_filt = evals1[np.abs(evals1) > threshold]
        evals2_filt = evals2[np.abs(evals2) > threshold]

        # Normalize to unit magnitude
        evals1_norm = evals1_filt / np.abs(evals1_filt)
        evals2_norm = evals2_filt / np.abs(evals2_filt)

        # For each eigenvalue in set 1, find closest match in set 2
        total_angle = 0
        n_matched = 0

        evals2_available = list(range(len(evals2_norm)))

        for ev1 in evals1_norm:
            if len(evals2_available) == 0:
                break

            # Compute angle to all available eigenvalues in set 2
            angles = []
            for idx in evals2_available:
                ev2 = evals2_norm[idx]
                # Angular distance on unit circle
                angle = np.abs(np.angle(ev1) - np.angle(ev2))
                # Wrap to [0, pi]
                angle = min(angle, 2 * np.pi - angle)
                angles.append(angle)

            # Find best match
            best_idx = np.argmin(angles)
            total_angle += angles[best_idx]
            evals2_available.pop(best_idx)
            n_matched += 1

        return total_angle / n_matched if n_matched > 0 else np.nan

    spectral_angle = compute_spectral_angle(evals_rnn, evals_theory)
    print(f"Custom spectral angle error: {np.degrees(spectral_angle):.4f}°")

    # Return results for visualization
    return {
        "W_hh_subspace": W_hh_subspace,
        "U_dominant": U_dominant,
        "singular_values": S,
        "variance_ratio": variance_ratio,
        "evals_rnn": evals_rnn,
        "evals_theory": evals_theory,
    }


def load_model(model_path, device):
    """Load checkpoint and create model with correct architecture."""
    print(f"Loading checkpoint from: {model_path}")

    # Load checkpoint to inspect hyperparameters
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    hparams = checkpoint["hyper_parameters"]

    # Extract model parameters from hyperparameters
    input_dim = hparams.get("input_dim", 8)
    hidden_dim = hparams.get("hidden_dim", 128)
    batch_size = hparams.get("batch_size", 64)

    print(
        f"Model config: input_dim={input_dim}, hidden_dim={hidden_dim}, batch_size={batch_size}"
    )

    # Create the model with the correct architecture
    model = RNNModel(input_dim, hidden_dim, input_dim, bias=False)

    # Load the checkpoint with the model
    lmodel = BinaryLinearVarVBTask.load_from_checkpoint(
        model_path, model=model, map_location=torch.device(device)
    )
    lmodel.eval()

    # Create EMT linear model for analysis
    emt_linear_model = LinearModel(lmodel.input_dim, lmodel.model.hidden_dim)
    emt_linear_model.parse_simple_rnn(lmodel.model)

    return lmodel, emt_linear_model, hparams, batch_size


def evaluate_model(lmodel, batch_size, test_horizon, device):
    """Run model evaluation on test data."""
    print(f"\n=== Evaluating Model ===")
    print(f"Test horizon: {test_horizon}")

    # Set test horizon and get sample
    lmodel.test_dataset.set_horizon(test_horizon)
    test_dataset = iter(lmodel.test_dataset)
    sample = next(test_dataset)
    x, y, seq_length = sample

    x = x.to(device)
    y = y.to(device)

    print(f"Sample sequence length: {seq_length}")
    print(f"Input shape: {x.shape}")

    # Initialize and run forward pass
    lmodel.model.initialize_hidden(batch_size=batch_size, device=device)
    y_hat = lmodel.model.forward(x)

    # Reshape predictions
    y_hat = y_hat.reshape((-1, lmodel.model.input_dim))
    y_hat = y_hat.reshape((-1, batch_size, lmodel.model.input_dim))
    y_hat = y_hat[seq_length:, :, :]
    y = y[seq_length:, :, :]

    # Convert to discrete predictions
    y_hat_predictions = y_hat.detach().clone()
    y_hat_predictions[y_hat_predictions >= 0] = 1
    y_hat_predictions[y_hat_predictions < 0] = -1
    y_hat_predictions = y_hat_predictions.long()

    # Move to CPU for analysis
    y_np = y.cpu().detach().numpy()
    y_hat_np = y_hat_predictions.cpu().detach().numpy()

    # Compute accuracy
    accuracy = (y_hat_np.astype(np.int_) == y_np.astype(np.int_)).astype(np.int_)
    mean_accuracy = np.mean(accuracy)

    print(f"Accuracy: {mean_accuracy:.4f}")

    return y_np, y_hat_np, seq_length, mean_accuracy


def construct_theoretical_solution(hparams, seq_length, input_dim, hidden_dim, lmodel):
    """Construct the theoretical Phi matrix for the task."""
    print(f"\n=== Constructing Theoretical Solution ===")

    task_dim = seq_length * input_dim
    print(
        f"Task dimension: {task_dim} (seq_length={seq_length} × input_dim={input_dim})"
    )

    # Build task-specific Phi
    phi_theoretical_small = np.eye(task_dim)
    phi_theoretical_small = np.roll(phi_theoretical_small, input_dim)
    phi_theoretical_small[:, :input_dim] = 0

    # Get f_operator for this sequence length
    task_id = hparams.get("task_id", (255, 0, 0))
    f_operator = lmodel.test_dataset.get_linear_operator(task_id, seq_length)
    phi_theoretical_small[-input_dim:, :] = f_operator.cpu().data.numpy().T

    # Embed in full hidden dimension space
    phi_theoretical_full = np.eye(hidden_dim)
    phi_theoretical_full[:task_dim, :task_dim] = phi_theoretical_small

    return phi_theoretical_small, phi_theoretical_full, task_dim


def plot_predictions(y_true, y_pred, output_dir):
    """Visualize predictions vs ground truth."""
    plt.figure(figsize=(10, 6))

    plt.subplot(211)
    plt.imshow(y_pred[:24, 0, :].T, cmap="coolwarm")
    plt.title("Prediction")
    plt.ylabel("Dimension")

    plt.subplot(212)
    plt.imshow(y_true[:24, 0, :].T, cmap="coolwarm")
    plt.title("Ground Truth")
    plt.xlabel("Time step")
    plt.ylabel("Dimension")

    plt.tight_layout()
    plt.savefig(osp.join(output_dir, "test_sample.png"), dpi=150)
    plt.close()
    print(f"Saved prediction visualization to {output_dir}/test_sample.png")


def plot_theoretical_matrices(all_results, seq_lengths, output_dir):
    """Visualize theoretical Phi matrices for all sequence lengths plus full space."""
    n_lengths = len(seq_lengths)

    # Create figure with n_lengths + 1 subplots (one for each seq length + full space)
    fig, axes = plt.subplots(1, n_lengths + 1, figsize=(5 * (n_lengths + 1), 5))

    # Plot phi_small for each sequence length
    for idx, seq_len in enumerate(seq_lengths):
        phi_small = all_results[seq_len]["phi_small"]
        task_dim = all_results[seq_len]["task_dim"]

        axes[idx].imshow(phi_small, cmap="coolwarm", vmin=-1, vmax=1)
        axes[idx].set_title(f"Φ (seq_len={seq_len})\n{task_dim}×{task_dim}")
        axes[idx].set_xlabel("Input dim")
        axes[idx].set_ylabel("Output dim")

    # Plot phi_full (same for all sequence lengths, so use any)
    phi_full = all_results[seq_lengths[0]]["phi_full"]
    hidden_dim = phi_full.shape[0]

    axes[n_lengths].imshow(phi_full, cmap="coolwarm", vmin=-1, vmax=1)
    axes[n_lengths].set_title(f"Φ (full space)\n{hidden_dim}×{hidden_dim}")
    axes[n_lengths].set_xlabel("Input dim")
    axes[n_lengths].set_ylabel("Output dim")

    fig.suptitle("Theoretical Solution Matrices", fontsize=14)
    fig.tight_layout()
    fig.savefig(osp.join(output_dir, "phi_theoretical.png"), dpi=150)

    plt.close()

    print(f"Saved theoretical matrices to {output_dir}/phi_theoretical.png")


def plot_eigenspectrum_full(W_hh, phi_full, output_dir):
    """Plot eigenspectra comparison in full hidden space."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    eig_vals_rnn, _ = np.linalg.eig(W_hh)
    ax.scatter(
        eig_vals_rnn.real, eig_vals_rnn.imag, label="RNN", marker="x", alpha=0.5, s=50
    )

    eig_vals_theory, _ = np.linalg.eig(phi_full)
    ax.scatter(
        eig_vals_theory.real,
        eig_vals_theory.imag,
        label="Theory",
        marker="o",
        alpha=0.5,
        s=30,
    )

    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title("Eigenspectrum - Full Hidden Space")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)

    unit_circle = patches.Circle(
        (0.0, 0.0), 1.0, fill=False, color="gray", linestyle="--"
    )
    ax.add_patch(unit_circle)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(osp.join(output_dir, "eigenspectrum_full.png"), dpi=150)
    plt.close()
    print(f"Saved full eigenspectrum to {output_dir}/eigenspectrum_full.png")


def plot_pca_analysis(pca_results, task_dim, output_dir):
    """Visualize PCA-based subspace analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Singular value spectrum
    ax = axes[0]
    ax.semilogy(pca_results["singular_values"], "o-", alpha=0.7)
    ax.axvline(task_dim, color="r", linestyle="--", label=f"Task dim = {task_dim}")
    ax.set_xlabel("Component index")
    ax.set_ylabel("Singular value")
    ax.set_title(
        f"Singular Values of W_hh\n(Top {task_dim} capture {100 * pca_results['variance_ratio']:.1f}% variance)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: PCA subspace eigenspectra comparison
    ax = axes[1]
    evals_rnn = pca_results["evals_rnn"]
    evals_theory = pca_results["evals_theory"]

    ax.scatter(
        evals_rnn.real,
        evals_rnn.imag,
        label="RNN (PCA subspace)",
        marker="x",
        alpha=0.6,
        s=50,
        color="C0",
    )
    ax.scatter(
        evals_theory.real,
        evals_theory.imag,
        label="Theory",
        marker="o",
        alpha=0.6,
        s=30,
        color="C1",
    )

    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title(f"Eigenspectrum in PCA Subspace ({task_dim}D)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)

    unit_circle = patches.Circle(
        (0.0, 0.0), 1.0, fill=False, color="gray", linestyle="--"
    )
    ax.add_patch(unit_circle)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(osp.join(output_dir, "pca_subspace_analysis.png"), dpi=150)
    plt.close()
    print(f"Saved PCA analysis to {output_dir}/pca_subspace_analysis.png")


def analyze_all_sequence_lengths(lmodel, emt_linear_model, hparams, output_dir):
    """Analyze PCA subspace for all sequence lengths the model was trained on."""
    print("Analyzing All Sequence Lengths")

    # Get sequence length range from hyperparameters
    seq_length_param = hparams.get("seq_length", 5)
    if isinstance(seq_length_param, tuple):
        seq_lengths = list(range(seq_length_param[0], seq_length_param[1] + 1))
    else:
        seq_lengths = [seq_length_param]

    print(f"Sequence lengths to analyze: {seq_lengths}")

    input_dim = lmodel.input_dim
    hidden_dim = emt_linear_model.W_hh.shape[0]

    # Store results for each sequence length
    all_results = {}

    for seq_len in seq_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---")

        # Construct theoretical solution for this sequence length
        phi_small, phi_full, task_dim = construct_theoretical_solution(
            hparams, seq_len, input_dim, hidden_dim, lmodel
        )

        # Run PCA-based subspace analysis
        pca_results = task_specific_subspace(emt_linear_model, task_dim, phi_small)

        all_results[seq_len] = {
            "phi_small": phi_small,
            "phi_full": phi_full,
            "task_dim": task_dim,
            "pca_results": pca_results,
        }

    return all_results, seq_lengths


def compute_empirical_phi_for_length(
    lmodel, emt_linear_model, hparams, seq_length, batch_size, device
):
    """
    Compute empirical Phi matrix for a specific sequence length.

    Args:
        lmodel: Lightning model
        emt_linear_model: EMT linear model
        hparams: Hyperparameters dictionary
        seq_length: Sequence length to compute Phi for
        batch_size: Batch size for forward pass
        device: Device to run on

    Returns:
        Phi_empirical: Empirical Phi matrix
        f_operator: Ground truth transition function
    """
    print(f"\nComputing empirical Phi for sequence length {seq_length}")

    # Set test horizon and get sample with specific sequence length
    # We need to run a forward pass to get hidden states for this sequence length
    lmodel.test_dataset.set_horizon(200)
    test_dataset = iter(lmodel.test_dataset)

    # Find a sample with the desired sequence length
    sample = None
    for _ in range(100):  # Try up to 100 samples
        candidate = next(test_dataset)
        x, y, sample_seq_length = candidate
        if sample_seq_length == seq_length:
            sample = candidate
            break

    if sample is None:
        print(
            f"Warning: Could not find sample with seq_length={seq_length}, using any available"
        )
        sample = candidate
        seq_length = sample_seq_length

    x, y, _ = sample
    x = x.to(device)

    # Run forward pass to get hidden states
    lmodel.model.initialize_hidden(batch_size=batch_size, device=device)
    _ = lmodel.model.forward(x)

    # Get hidden states from the forward pass
    hidden_states = lmodel.model.all_hidden.cpu().data.numpy()[:, 0, :].squeeze()

    # Get f_operator for this sequence length
    task_id = hparams.get("task_id", (255, 0, 0))
    f_operator = lmodel.test_dataset.get_linear_operator(task_id, seq_length)
    f_operator_np = f_operator.cpu().data.numpy().T

    # Compute variable basis
    Psi, Psi_star = emt_linear_model.get_variable_basis(
        seq_length,
        hidden_states,
        alpha=1,
        f_operator=f_operator_np,
        strength=1,
        threshold=0.99,
    )

    # Compute empirical Phi in the space of variable memories
    Phi_empirical = Psi_star @ emt_linear_model.W_hh @ Psi

    return Phi_empirical, f_operator_np


def plot_empirical_phi_all_lengths(
    lmodel, emt_linear_model, hparams, seq_lengths, batch_size, device, output_dir
):
    """
    Compute and visualize empirical Phi matrices for all sequence lengths.

    Args:
        lmodel: Lightning model
        emt_linear_model: EMT linear model
        hparams: Hyperparameters dictionary
        seq_lengths: List of sequence lengths
        batch_size: Batch size for forward pass
        device: Device to run on
        output_dir: Directory to save plots
    """
    print("\n=== Computing Empirical Phi for All Sequence Lengths ===")

    n_lengths = len(seq_lengths)
    empirical_phis = {}
    f_operators = {}

    # Compute empirical Phi for each sequence length
    for seq_len in seq_lengths:
        phi_emp, f_op = compute_empirical_phi_for_length(
            lmodel, emt_linear_model, hparams, seq_len, batch_size, device
        )
        empirical_phis[seq_len] = phi_emp
        f_operators[seq_len] = f_op

    # Create visualization with all empirical Phi matrices in one row
    fig, axes = plt.subplots(1, n_lengths, figsize=(6 * n_lengths, 6))
    if n_lengths == 1:
        axes = [axes]

    for idx, seq_len in enumerate(seq_lengths):
        ax = axes[idx]
        phi_emp = empirical_phis[seq_len]
        input_dim = lmodel.input_dim

        im = ax.imshow(phi_emp.real, cmap="coolwarm", vmin=-1, vmax=1, alpha=1)

        # Add grid lines to separate variable blocks
        for i in range(input_dim):
            ax.axhline(y=((i + 1) * seq_len) - 0.9, color="w", linestyle="-", lw=0.7)

        for j in range(input_dim):
            ax.axvline(x=((j + 1) * seq_len) - 0.5, color="w", linestyle="-", lw=0.7)

        ax.axhline(y=input_dim * seq_len - 0.5, color="k", linestyle="-")
        ax.axvline(x=input_dim * seq_len - 0.5, color="k", linestyle="-")

        # Set ticks to show variable indices
        ax.set_xticks(
            np.arange(
                input_dim // 2,
                input_dim * seq_len,
                input_dim,
            ),
            range(1, seq_len + 1),
        )
        ax.set_yticks(
            np.arange(
                input_dim // 2,
                input_dim * seq_len,
                input_dim,
            ),
            range(1, seq_len + 1),
        )
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.set_title(f"Φ empirical (seq_len={seq_len})")

        # Add colorbar for each subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

    fig.suptitle("Empirical Phi Matrices (Variable Memory Space)", fontsize=14)
    fig.tight_layout()
    fig.savefig(osp.join(output_dir, "phi_empirical_all_lengths.png"), dpi=150)
    plt.close()

    print(f"Saved empirical Phi matrices to {output_dir}/phi_empirical_all_lengths.png")

    # Also plot the transition functions
    fig, axes = plt.subplots(1, n_lengths, figsize=(5 * n_lengths, 5))
    if n_lengths == 1:
        axes = [axes]

    for idx, seq_len in enumerate(seq_lengths):
        ax = axes[idx]
        f_op = f_operators[seq_len]

        im = ax.imshow(f_op, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title(f"f_operator (seq_len={seq_len})")
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

    fig.suptitle("Transition Functions (Ground Truth)", fontsize=14)
    fig.tight_layout()
    fig.savefig(osp.join(output_dir, "transition_func_all_lengths.png"), dpi=150)
    plt.close()

    print(f"Saved transition functions to {output_dir}/transition_func_all_lengths.png")


def plot_multi_length_comparison(all_results, seq_lengths, output_dir):
    """Create comprehensive visualization comparing all sequence lengths."""
    n_lengths = len(seq_lengths)

    # Create a large figure with subplots for each sequence length
    fig, axes = plt.subplots(2, n_lengths, figsize=(6 * n_lengths, 10))
    if n_lengths == 1:
        axes = axes.reshape(2, 1)

    for idx, seq_len in enumerate(seq_lengths):
        results = all_results[seq_len]
        pca_results = results["pca_results"]
        task_dim = results["task_dim"]

        # Row 1: Singular value spectrum
        ax = axes[0, idx]
        ax.semilogy(pca_results["singular_values"], "o-", alpha=0.7, markersize=3)
        ax.axvline(
            task_dim,
            color="r",
            linestyle="--",
            label=f"Task dim = {task_dim}",
            linewidth=2,
        )
        ax.set_xlabel("Component index")
        ax.set_ylabel("Singular value")
        ax.set_title(
            f"Seq Length {seq_len}\n({100 * pca_results['variance_ratio']:.1f}% variance)"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Row 2: Eigenspectrum in PCA subspace
        ax = axes[1, idx]
        evals_rnn = pca_results["evals_rnn"]
        evals_theory = pca_results["evals_theory"]

        ax.scatter(
            evals_rnn.real,
            evals_rnn.imag,
            label="RNN",
            marker="x",
            alpha=0.6,
            s=50,
            color="C0",
        )
        ax.scatter(
            evals_theory.real,
            evals_theory.imag,
            label="Theory",
            marker="o",
            alpha=0.6,
            s=30,
            color="C1",
        )

        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.set_title(f"PCA Subspace ({task_dim}D)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)

        unit_circle = patches.Circle(
            (0.0, 0.0), 1.0, fill=False, color="gray", linestyle="--", linewidth=1
        )
        ax.add_patch(unit_circle)
        ax.set_aspect("equal")

        # Set consistent axis limits across all plots
        max_val = max(np.abs(evals_rnn).max(), np.abs(evals_theory).max()) * 1.1
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)

    fig.suptitle("PCA Subspace Analysis Across Sequence Lengths", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(osp.join(output_dir, "multi_length_pca_analysis.png"), dpi=150)
    plt.close()
    print(
        f"\nSaved multi-length comparison to {output_dir}/multi_length_pca_analysis.png"
    )


def main():
    """Main evaluation pipeline."""

    # Configuration
    ACCELERATOR = "gpu"
    TEST_HORIZON = 200
    logs_dir = osp.join("rcopy_test", "lightning_logs")
    ckpt_name = osp.join("version_14", "checkpoints", "epoch=49-step=100000.ckpt")
    model_path = osp.join(logs_dir, ckpt_name)
    output_dir = osp.join("evaluate", ckpt_name)

    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if ACCELERATOR == "gpu" else "cpu"

    print("RNN Variable Binding Task Evaluation")

    # 1. Load model
    lmodel, emt_linear_model, hparams, batch_size = load_model(model_path, device)

    # 2. Evaluate on test data
    y_true, y_pred, seq_length, accuracy = evaluate_model(
        lmodel, batch_size, TEST_HORIZON, device
    )

    # 3. Construct theoretical solution for the sampled sequence length
    phi_small, phi_full, task_dim = construct_theoretical_solution(
        hparams, seq_length, lmodel.input_dim, emt_linear_model.W_hh.shape[0], lmodel
    )

    # 4. Analyze ALL sequence lengths the model was trained on
    all_results, seq_lengths = analyze_all_sequence_lengths(
        lmodel, emt_linear_model, hparams, output_dir
    )

    # 5. Generate visualizations
    plot_predictions(y_true, y_pred, output_dir)
    plot_theoretical_matrices(all_results, seq_lengths, output_dir)
    plot_eigenspectrum_full(emt_linear_model.W_hh, phi_full, output_dir)

    # 6. PCA-based subspace analysis for sampled sequence
    pca_results = task_specific_subspace(emt_linear_model, task_dim, phi_small)
    plot_pca_analysis(pca_results, task_dim, output_dir)

    # 7. Multi-length comparison visualization
    plot_multi_length_comparison(all_results, seq_lengths, output_dir)

    # 8. Empirical Phi for all sequence lengths
    plot_empirical_phi_all_lengths(
        lmodel, emt_linear_model, hparams, seq_lengths, batch_size, device, output_dir
    )

    print(f"Results saved to: {output_dir}")
    print(f"Analyzed sequence lengths: {seq_lengths}")


if __name__ == "__main__":
    main()
