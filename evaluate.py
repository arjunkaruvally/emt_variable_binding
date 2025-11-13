"""
Evaluate RNN models trained on the repeat copy task.
Supports evaluation of single models or multiple models.
Saves plots to respective output folders like: 50_models_seqlen_8/model-38/results

Usage:
    # Evaluate all models in a directory (loads config.json automatically)
    python evaluate.py --models_dir 50_models_seqlen_8

    # Evaluate with explicit sequence length
    python evaluate.py --models_dir 50_models_seqlen_8 --seq_length 8

    # Evaluate models with variable sequence lengths (range)
    python evaluate.py --models_dir my_models --seq_length 3 7

    # Evaluate a single model by specifying its pattern
    python evaluate.py --models_dir 50_models_seqlen_8 --model_pattern "model-42"

    # Evaluate on CPU instead of GPU
    python evaluate.py --models_dir my_models --accelerator cpu

    # Customize test horizon
    python evaluate.py --models_dir my_models --test_horizon 500

Notes:
    - The models_dir should contain model subdirectories (e.g., model-0, model-1, ...)
    - Each model subdirectory should have: lightning_logs/version_*/checkpoints/*.ckpt
    - If config.json exists in models_dir, sequence lengths are loaded automatically
    - Use --model_pattern to match specific models (default: "model-*")
    - Results are saved to: {models_dir}/{model_name}/results/
"""

import argparse
import glob
import json
import os
import os.path as osp
from datetime import datetime

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from emt_tools.models.linearModel import LinearModel
from emt_tools.utils import spectral_comparison
from mpl_toolkits.axes_grid1 import make_axes_locatable

from em_discrete.models.rnn_model import RNNModel
from em_discrete.tasks.variable_linear_vb import BinaryLinearVarVBTask

# Disable LaTeX rendering globally - use matplotlib's mathtext instead
# plt.rcParams["text.usetex"] = True


def task_specific_subspace(linear_model, task_dim, phi_theoretical):
    """
    Use PCA/SVD to identify the dominant subspace of W_hh and project
    the theoretical matrix into that subspace for comparison.
    """
    W_hh = linear_model.W_hh

    print("\n=== PCA-based Subspace Analysis ===")
    print(f"Full W_hh shape: {W_hh.shape}")
    print(f"Task dimension: {task_dim}")

    # Perform SVD on W_hh to find principal components
    U, S, Vh = np.linalg.svd(W_hh, full_matrices=True)

    # Analyze singular values to see how much variance is in top task_dim dimensions
    total_variance = np.sum(S**2)
    top_k_variance = np.sum(S[:task_dim] ** 2)
    variance_ratio = top_k_variance / total_variance

    print("\nSingular value analysis:")
    print(
        f"Top {task_dim} singular values capture {100 * variance_ratio:.2f}% of variance"
    )
    print(f"Top 10 singular values: {S[:10]}")
    print(
        f"Singular values {task_dim - 5} to {task_dim + 5}: {S[max(0, task_dim - 5) : task_dim + 5]}"
    )

    # Extract the top task_dim principal components
    U_dominant = U[:, :task_dim]  # First task_dim left singular vectors

    # Project W_hh into the dominant subspace
    W_hh_subspace = U_dominant.T @ W_hh @ U_dominant

    print("\nProjected matrices:")
    print(f"W_hh_subspace shape: {W_hh_subspace.shape}")
    print(f"phi_theoretical shape: {phi_theoretical.shape}")

    # Compute eigenvalues for diagnostics
    evals_rnn = np.linalg.eigvals(W_hh_subspace)
    evals_theory = np.linalg.eigvals(phi_theoretical)

    print("\nEigenvalue diagnostics:")
    print(
        f"RNN (PCA subspace) - count: {len(evals_rnn)}, |�|>0.01: {np.sum(np.abs(evals_rnn) > 0.01)}"
    )
    print(
        f"Theory eigenvalues - count: {len(evals_theory)}, |�|>0.01: {np.sum(np.abs(evals_theory) > 0.01)}"
    )

    # Try the built-in spectral comparison
    try:
        spectral_error = spectral_comparison(W_hh_subspace, phi_theoretical)
        if not np.isnan(spectral_error):
            print(f"\nSpectral error (in degrees): {np.degrees(spectral_error):.4f}")
        else:
            print("\nSpectral comparison returned NaN (numerical issues)")
    except Exception as e:
        print(f"\nSpectral comparison failed: {e}")
        spectral_error = np.nan

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

    # Return results for visualization
    return {
        "W_hh_subspace": W_hh_subspace,
        "U_dominant": U_dominant,
        "singular_values": S,
        "variance_ratio": variance_ratio,
        "evals_rnn": evals_rnn,
        "evals_theory": evals_theory,
        "spectral_error": spectral_error,
        "frob_error": frob_error,
        "frob_relative": frob_relative,
        "eigenvalue_error": eigenvalue_error,
    }


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


def evaluate_model(lmodel, batch_size, test_horizon, device):
    """Run model evaluation on test data."""
    print("\n=== Evaluating Model ===")

    # Check if this is an infinite horizon model
    from em_discrete.tasks.infinite_horizon_vb import BinaryLinearVBInfiniteHorizonTask

    is_infinite_horizon = isinstance(lmodel, BinaryLinearVBInfiniteHorizonTask)

    if is_infinite_horizon:
        print("Infinite horizon mode: Direct evaluation")
        # For infinite horizon, evaluate directly without set_horizon
        lmodel.eval()
        lmodel = lmodel.to(device)  # Move model to correct device

        test_dataset = iter(lmodel.test_dataloader())
        sample = next(test_dataset)

        # Get data
        x, y, seq_lengths = sample
        x = x.to(device)
        y = y.to(device)

        print(f"Input shape (raw): {x.shape}")
        print(f"Target shape (raw): {y.shape}")

        # Infinite horizon data has shape (num_sequences, time, batch, features)
        # We need (time, batch, features) for the RNN
        if x.dim() == 4:
            x = x.squeeze(0)  # Remove sequence dimension
            y = y.squeeze(0)

        print(f"Input shape (processed): {x.shape}")
        print(f"Target shape (processed): {y.shape}")

        # Run forward pass
        lmodel.model.initialize_hidden(batch_size=batch_size, device=device)
        with torch.no_grad():
            y_hat = lmodel.model.forward(x)

        # For infinite horizon, we evaluate over the full sequence
        # The model outputs are for data dims only (not including EOS)
        output_dim = lmodel.input_dim
        y_hat = y_hat.reshape((-1, output_dim))
        y_hat = y_hat.reshape((-1, batch_size, output_dim))

        # Don't slice - use full sequence for infinite horizon
        # The targets y are already properly structured

        # Identify EOS tokens (all -1s pattern in input x)
        # EOS is encoded as all -1s across all dimensions
        is_eos = torch.all(x == -1, dim=-1)  # (time, batch)

        # Take only timesteps where we have targets (y != 0)
        # In infinite horizon, y is zero during input phases
        has_target = torch.any(y != 0, dim=-1)  # (time, batch)

        # Exclude EOS timesteps from evaluation
        should_evaluate = has_target & (
            ~is_eos
        )  # Only evaluate output timesteps that are NOT EOS

        # Convert to discrete predictions
        y_hat_predictions = y_hat.detach().clone()
        y_hat_predictions[y_hat_predictions >= 0] = 1
        y_hat_predictions[y_hat_predictions < 0] = -1
        y_hat_predictions = y_hat_predictions.long()

        # Move to CPU
        y_np = y.cpu().detach().numpy()
        y_hat_np = y_hat_predictions.cpu().detach().numpy()
        should_evaluate_np = should_evaluate.cpu().detach().numpy()
        is_eos_np = is_eos.cpu().detach().numpy()

        # Compute accuracy only where we should evaluate (output timesteps, excluding EOS)
        correct = (y_hat_np == y_np).astype(np.int_)
        # Expand should_evaluate to all dimensions
        should_evaluate_expanded = should_evaluate_np[:, :, np.newaxis]

        # Count correct predictions only where we should evaluate
        n_correct = (correct * should_evaluate_expanded).sum()
        n_total = should_evaluate_expanded.sum() * output_dim
        mean_accuracy = n_correct / n_total if n_total > 0 else 0.0

        print(f"Accuracy (on output timesteps, excluding EOS): {mean_accuracy:.4f}")
        print(
            f"Evaluated timesteps: {should_evaluate_np.sum()} / {should_evaluate_np.size} total timesteps"
        )
        print(f"EOS timesteps excluded: {is_eos_np.sum()}")

        # For infinite horizon, return a representative sequence length
        # seq_lengths can be a complex nested structure - flatten and find a valid int
        def extract_scalar(obj):
            """Recursively extract a scalar integer from nested structures."""
            if isinstance(obj, int):
                return obj
            if torch.is_tensor(obj):
                if obj.numel() == 1:
                    return int(obj.item())
                elif obj.numel() > 1:
                    return extract_scalar(obj[0])
            if isinstance(obj, (list, tuple)) and len(obj) > 0:
                for item in obj:
                    result = extract_scalar(item)
                    if result is not None and result > 0:
                        return result
            return None

        seq_length = extract_scalar(seq_lengths)
        if seq_length is None or seq_length == 0:
            seq_length = 8  # Default fallback

        # Return the input data for visualization (EOS encoded as all -1s within data)
        x_np = x.cpu().detach().numpy()
        return y_np, y_hat_np, seq_length, mean_accuracy, is_eos_np, x_np
    else:
        # Regular mode evaluation
        print(f"Regular mode: Test horizon: {test_horizon}")
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

        # Reshape predictions (use lmodel.input_dim which is the data dimension, not model input)
        output_dim = (
            lmodel.input_dim
        )  # This is the data dimension (e.g., 8), not including EOS
        y_hat = y_hat.reshape((-1, output_dim))
        y_hat = y_hat.reshape((-1, batch_size, output_dim))
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

        return y_np, y_hat_np, seq_length, mean_accuracy, None, None


def construct_theoretical_solution(hparams, seq_length, input_dim, hidden_dim, lmodel):
    """Construct the theoretical Phi matrix for the task."""
    print("\n=== Constructing Theoretical Solution ===")

    task_dim = seq_length * input_dim
    print(
        f"Task dimension: {task_dim} (seq_length={seq_length} � input_dim={input_dim})"
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


def plot_predictions(y_true, y_pred, output_dir, is_eos=None, input_data=None):
    """Visualize predictions vs ground truth with error highlighting.

    Args:
        y_true: Ground truth targets (time, batch, features)
        y_pred: Model predictions (time, batch, features)
        output_dir: Directory to save the plot
        is_eos: Boolean mask for EOS timesteps (time, batch). If provided, EOS timesteps are marked in gray.
        input_data: Input data (time, batch, features). If provided, displays input alongside predictions/targets.
    """
    fig = plt.figure(figsize=(12, 9))

    # Show first batch sample, first 40 timesteps
    n_timesteps = min(40, y_pred.shape[0])

    # Just use the regular data (EOS encoded within, same dimensions as output)
    y_pred_display = y_pred[:n_timesteps, 0, :].T
    y_true_display = y_true[:n_timesteps, 0, :].T

    # Prediction
    ax1 = plt.subplot(311)
    im1 = ax1.imshow(y_pred_display, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax1.set_title("Prediction", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Dimension", fontsize=10)
    ax1.set_xticks(range(0, n_timesteps, 5))
    plt.colorbar(im1, ax=ax1, label="Value")

    # Mark EOS timesteps if provided
    if is_eos is not None:
        eos_times = np.where(is_eos[:n_timesteps, 0])[0]
        for t in eos_times:
            ax1.axvline(t, color="black", alpha=0.5, linewidth=1.5, linestyle="--")

    # Ground Truth
    ax2 = plt.subplot(312)
    im2 = ax2.imshow(y_true_display, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax2.set_title("Ground Truth", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Dimension", fontsize=10)
    ax2.set_xticks(range(0, n_timesteps, 5))
    plt.colorbar(im2, ax=ax2, label="Value")

    # Mark EOS timesteps if provided
    if is_eos is not None:
        for t in eos_times:
            ax2.axvline(t, color="black", alpha=0.5, linewidth=1.5, linestyle="--")

    # Error map (1 = error, 0 = correct, 0.5 = EOS/not evaluated)
    ax3 = plt.subplot(313)
    errors_original = (y_pred[:n_timesteps, 0, :] != y_true[:n_timesteps, 0, :]).astype(
        float
    )

    # Compute accuracy excluding EOS (use original errors before visualization modification)
    if is_eos is not None:
        # Only count errors where is_eos is False
        non_eos_mask = ~is_eos[:n_timesteps, 0, np.newaxis]  # (time, 1)
        # Count errors only at non-EOS positions
        non_eos_errors = errors_original * non_eos_mask
        n_errors = int(non_eos_errors.sum())
        n_total = int(non_eos_mask.sum() * y_pred.shape[2])
        sample_accuracy = 1.0 - (n_errors / n_total) if n_total > 0 else 0.0

        # For visualization: mark EOS timesteps as gray (0.5)
        eos_mask = is_eos[:n_timesteps, 0, np.newaxis]  # (time, 1)
        errors_display = np.where(eos_mask, 0.5, errors_original)  # EOS = gray (0.5)

        title = f"Errors: Red=Wrong, White=Correct, Gray=EOS (excluded) | Acc: {sample_accuracy:.4f} ({n_errors}/{n_total} non-EOS)"
    else:
        errors_display = errors_original
        sample_accuracy = 1.0 - errors_original.mean()
        n_errors = int(errors_original.sum())
        title = f"Errors (Red = Wrong, White = Correct) | Accuracy: {sample_accuracy:.4f} ({n_errors}/{errors_original.size} errors)"

    # Use custom colormap: white=correct(0), gray=EOS(0.5), red=error(1)
    from matplotlib.colors import ListedColormap

    colors = ["white", "lightgray", "red"]
    cmap = ListedColormap(colors)

    im3 = ax3.imshow(errors_display.T, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax3.set_title(title, fontsize=12, fontweight="bold")
    ax3.set_xlabel("Time step", fontsize=10)
    ax3.set_ylabel("Dimension", fontsize=10)
    ax3.set_xticks(range(0, n_timesteps, 5))

    # Mark EOS timesteps with vertical lines
    if is_eos is not None:
        for t in eos_times:
            ax3.axvline(t, color="black", alpha=0.5, linewidth=1, linestyle="--")

    cbar = plt.colorbar(im3, ax=ax3, ticks=[0, 0.5, 1])
    if is_eos is not None:
        cbar.ax.set_yticklabels(["Correct", "EOS", "Error"])
    else:
        cbar.ax.set_yticklabels(["Correct", "", "Error"])

    plt.tight_layout()
    plt.savefig(osp.join(output_dir, "test_sample.png"), dpi=150)
    plt.close()
    print(f"Saved prediction visualization to {output_dir}/test_sample.png")


def plot_theoretical_matrices(all_results, seq_lengths, output_dir):
    """Visualize theoretical Phi matrices for all sequence lengths plus full space."""
    n_lengths = len(seq_lengths)

    # Create figure with n_lengths + 1 subplots
    fig, axes = plt.subplots(1, n_lengths + 1, figsize=(5 * (n_lengths + 1), 5))

    # Plot phi_small for each sequence length
    for idx, seq_len in enumerate(seq_lengths):
        phi_small = all_results[seq_len]["phi_small"]
        task_dim = all_results[seq_len]["task_dim"]

        axes[idx].imshow(phi_small, cmap="coolwarm", vmin=-1, vmax=1)
        axes[idx].set_title(rf"$\Phi$ (seq_len={seq_len})" + f"\n{task_dim}" + r"$\times$" + f"{task_dim}")
        axes[idx].set_xlabel("Input dim")
        axes[idx].set_ylabel("Output dim")

    # Plot phi_full
    phi_full = all_results[seq_lengths[0]]["phi_full"]
    hidden_dim = phi_full.shape[0]

    axes[n_lengths].imshow(phi_full, cmap="coolwarm", vmin=-1, vmax=1)
    axes[n_lengths].set_title(rf"$\Phi$ (full space)" + f"\n{hidden_dim}" + r"$\times$" + f"{hidden_dim}")
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

    ax.set_xlabel(r"$\mathrm{Re}(\lambda)$")
    ax.set_ylabel(r"$\mathrm{Im}(\lambda)$")
    ax.set_title(r"Eigenspectrum - Full Hidden Space $W_{hh}$")
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
        r"Singular Values of $W_{{hh}}$"
        + f"\n(Top {task_dim} capture {100 * pca_results['variance_ratio']:.1f}% variance)"
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

    ax.set_xlabel(r"$\mathrm{Re}(\lambda)$")
    ax.set_ylabel(r"$\mathrm{Im}(\lambda)$")
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


def analyze_all_sequence_lengths(
    lmodel, emt_linear_model, hparams, output_dir, seq_lengths_override=None
):
    """Analyze PCA subspace for all sequence lengths the model was trained on.

    Args:
        lmodel: Lightning model
        emt_linear_model: EMT linear model
        hparams: Hyperparameters dictionary
        output_dir: Output directory for plots
        seq_lengths_override: Optional list of sequence lengths to analyze (overrides hparams)
    """
    print("Analyzing All Sequence Lengths")

    # Get sequence length range
    if seq_lengths_override is not None:
        seq_lengths = seq_lengths_override
        print(f"Using override sequence lengths: {seq_lengths}")
    else:
        # Get from hyperparameters
        seq_length_param = hparams.get("seq_length", 5)
        if isinstance(seq_length_param, tuple):
            seq_lengths = list(range(seq_length_param[0], seq_length_param[1] + 1))
        else:
            seq_lengths = [seq_length_param]
        print(f"Sequence lengths from hyperparameters: {seq_lengths}")

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

        ax.set_xlabel(r"$\mathrm{Re}(\lambda)$")
        ax.set_ylabel(r"$\mathrm{Im}(\lambda)$")
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


def compute_empirical_phi_for_length(
    lmodel, emt_linear_model, hparams, seq_length, batch_size, device
):
    """
    Compute empirical Phi matrix for a specific sequence length.
    """
    print(f"\nComputing empirical Phi for sequence length {seq_length}")

    # Check if infinite horizon model
    from em_discrete.tasks.infinite_horizon_vb import BinaryLinearVBInfiniteHorizonTask

    is_infinite_horizon = isinstance(lmodel, BinaryLinearVBInfiniteHorizonTask)

    if is_infinite_horizon:
        # For infinite horizon, use test_dataloader directly
        test_dataset = iter(lmodel.test_dataloader())
        sample = next(test_dataset)
        x, y, seq_lengths_list = sample

        # Handle 4D tensor from infinite horizon
        if x.dim() == 4:
            x = x.squeeze(0)
            y = y.squeeze(0)

        x = x.to(device)
    else:
        # Regular model: Set test horizon and get sample with specific sequence length
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
        ax.set_title(rf"$\Phi$ empirical (seq_len={seq_len})")

        # Add colorbar for each subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

    fig.suptitle(
        r"Empirical $\Phi$ Matrices (Variable Memory Space $\Psi$)", fontsize=14
    )
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
        ax.set_title(rf"$f$ operator (seq_len={seq_len})")
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

    fig.suptitle(r"Transition Functions $f$ (Ground Truth)", fontsize=14)
    fig.tight_layout()
    fig.savefig(osp.join(output_dir, "transition_func_all_lengths.png"), dpi=150)
    plt.close()

    print(f"Saved transition functions to {output_dir}/transition_func_all_lengths.png")

    # Return the empirical Phis for further analysis
    return empirical_phis, f_operators


def plot_variable_basis_eigenspectra(
    empirical_phis, f_operators, seq_lengths, lmodel, output_dir
):
    """
    Plot eigenvalue spectra using the variable basis method.
    Compares empirical Phi eigenvalues with theoretical eigenvalues.
    """
    print("\n=== Plotting Variable Basis Eigenspectra ===")

    n_lengths = len(seq_lengths)
    input_dim = lmodel.input_dim

    # Create figure with subplots for each sequence length (3 rows now)
    fig, axes = plt.subplots(3, n_lengths, figsize=(6 * n_lengths, 14))
    if n_lengths == 1:
        axes = axes.reshape(3, 1)

    for idx, seq_len in enumerate(seq_lengths):
        phi_emp = empirical_phis[seq_len]
        f_op = f_operators[seq_len]

        # Construct theoretical Phi for this sequence length
        task_dim = seq_len * input_dim
        phi_theoretical = np.eye(task_dim)
        phi_theoretical = np.roll(phi_theoretical, input_dim)  # NO axis parameter!
        phi_theoretical[:, :input_dim] = 0  # Set first input_dim COLUMNS to 0
        phi_theoretical[-input_dim:, :] = f_op

        # Compute eigenvalues
        evals_empirical = np.linalg.eigvals(phi_emp)
        evals_theoretical = np.linalg.eigvals(phi_theoretical)

        # Row 1: Eigenspectrum scatter plot
        ax = axes[0, idx]
        ax.scatter(
            evals_empirical.real,
            evals_empirical.imag,
            label="Empirical (Variable Basis)",
            marker="x",
            alpha=0.6,
            s=80,
            color="C0",
            linewidths=2,
        )
        ax.scatter(
            evals_theoretical.real,
            evals_theoretical.imag,
            label="Theoretical",
            marker="o",
            alpha=0.6,
            s=50,
            color="C1",
        )

        ax.set_xlabel(r"$\mathrm{Re}(\lambda)$", fontsize=10)
        ax.set_ylabel(r"$\mathrm{Im}(\lambda)$", fontsize=10)
        ax.set_title(
            r"Variable Basis ($\Psi$) Eigenspectrum"
            + f"\n(seq_len={seq_len}, dim={task_dim})"
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)

        # Add unit circle
        import matplotlib.patches as patches

        unit_circle = patches.Circle(
            (0.0, 0.0), 1.0, fill=False, color="gray", linestyle="--", linewidth=1
        )
        ax.add_patch(unit_circle)
        ax.set_aspect("equal")

        # Set consistent axis limits
        max_val = (
            max(np.abs(evals_empirical).max(), np.abs(evals_theoretical).max()) * 1.1
        )
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)

        # Row 2: Eigenvalue magnitude histogram
        ax = axes[1, idx]
        mag_empirical = np.abs(evals_empirical)
        mag_theoretical = np.abs(evals_theoretical)

        bins = np.linspace(0, max(mag_empirical.max(), mag_theoretical.max()) * 1.1, 30)
        ax.hist(
            mag_empirical,
            bins=bins,
            alpha=0.6,
            label="Empirical",
            color="C0",
            edgecolor="black",
        )
        ax.hist(
            mag_theoretical,
            bins=bins,
            alpha=0.6,
            label="Theoretical",
            color="C1",
            edgecolor="black",
        )

        ax.axvline(
            1.0, color="red", linestyle="--", linewidth=2, label=r"$|\lambda|=1$"
        )
        ax.set_xlabel(r"Eigenvalue Magnitude $|\lambda|$", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"Magnitude Distribution (seq_len={seq_len})")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

        # Row 3: Eigenvalue angle distribution (phase in complex plane)
        ax = axes[2, idx]
        # Compute angles (phase/argument) in radians
        angle_empirical = np.angle(evals_empirical)  # Returns values in [-π, π]
        angle_theoretical = np.angle(evals_theoretical)

        bins = np.linspace(-np.pi, np.pi, 40)
        ax.hist(
            angle_empirical,
            bins=bins,
            alpha=0.6,
            label="Empirical",
            color="C0",
            edgecolor="black",
        )
        ax.hist(
            angle_theoretical,
            bins=bins,
            alpha=0.6,
            label="Theoretical",
            color="C1",
            edgecolor="black",
        )

        ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel(r"Eigenvalue Angle $\theta = \arg(\lambda)$ (radians)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"Angle Distribution (seq_len={seq_len})")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-np.pi, np.pi)
        # Add tick marks at multiples of π/2
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

        # Print statistics
        print(f"\nSeq length {seq_len}:")
        print(
            f"  Empirical: |λ| ∈ [{mag_empirical.min():.3f}, {mag_empirical.max():.3f}]"
        )
        print(
            f"  Theoretical: |λ| ∈ [{mag_theoretical.min():.3f}, {mag_theoretical.max():.3f}]"
        )
        print(
            f"  Empirical eigenvalues with |λ| > 0.9: {np.sum(mag_empirical > 0.9)}/{len(mag_empirical)}"
        )
        print(
            f"  Theoretical eigenvalues with |λ| > 0.9: {np.sum(mag_theoretical > 0.9)}/{len(mag_theoretical)}"
        )

    fig.suptitle(
        r"Variable Basis ($\Psi^\star W_{hh} \Psi$) Eigenvalue Analysis",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(osp.join(output_dir, "variable_basis_eigenspectra.png"), dpi=150)
    plt.close()

    print(
        f"\nSaved variable basis eigenspectra to {output_dir}/variable_basis_eigenspectra.png"
    )


def plot_eigenvalue_angle_distribution(all_results, seq_lengths, output_dir):
    """
    Plot the distribution of eigenvalue angles (phase/argument) for RNN and theoretical matrices.
    Shows histograms with angle theta on the x-axis.
    """
    print("\n=== Plotting Eigenvalue Angle Distribution ===")

    n_lengths = len(seq_lengths)

    # Create figure with subplots for each sequence length
    fig, axes = plt.subplots(1, n_lengths, figsize=(6 * n_lengths, 5))
    if n_lengths == 1:
        axes = [axes]

    for idx, seq_len in enumerate(seq_lengths):
        ax = axes[idx]
        pca_results = all_results[seq_len]["pca_results"]

        evals_rnn = pca_results["evals_rnn"]
        evals_theory = pca_results["evals_theory"]

        # Compute angles (phase/argument) in radians
        angle_rnn = np.angle(evals_rnn)  # Returns values in [-π, π]
        angle_theory = np.angle(evals_theory)

        # Create histogram
        bins = np.linspace(-np.pi, np.pi, 40)

        ax.hist(
            angle_rnn, bins=bins, alpha=0.6, label="RNN", color="C0", edgecolor="black"
        )
        ax.hist(
            angle_theory,
            bins=bins,
            alpha=0.6,
            label="Theory",
            color="C1",
            edgecolor="black",
        )

        # Add vertical lines at key angles
        ax.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.axvline(np.pi / 2, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.axvline(-np.pi / 2, color="gray", linestyle=":", linewidth=1, alpha=0.5)

        ax.set_xlabel(r"Eigenvalue Angle $\theta$ (radians)")
        ax.set_ylabel("Count")
        ax.set_title(f"Eigenvalue Angle Distribution\n(seq_len={seq_len})")
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        r"Distribution of Eigenvalue Angles $\arg(\lambda)$ (Phase)", fontsize=14
    )
    fig.tight_layout()
    fig.savefig(osp.join(output_dir, "eigenvalue_angle_distribution.png"), dpi=150)
    plt.close()

    print(
        f"Saved eigenvalue angle distribution to {output_dir}/eigenvalue_angle_distribution.png"
    )


def plot_eigenvalue_magnitude_distribution(all_results, seq_lengths, output_dir):
    """
    Plot the distribution of eigenvalue magnitudes (radius) for RNN and theoretical matrices.
    Shows histograms with radius (|λ|) on the x-axis.
    """
    print("\n=== Plotting Eigenvalue Magnitude Distribution ===")

    n_lengths = len(seq_lengths)

    # Create figure with subplots for each sequence length
    fig, axes = plt.subplots(1, n_lengths, figsize=(6 * n_lengths, 5))
    if n_lengths == 1:
        axes = [axes]

    for idx, seq_len in enumerate(seq_lengths):
        ax = axes[idx]
        pca_results = all_results[seq_len]["pca_results"]

        evals_rnn = pca_results["evals_rnn"]
        evals_theory = pca_results["evals_theory"]

        # Compute magnitudes
        mag_rnn = np.abs(evals_rnn)
        mag_theory = np.abs(evals_theory)

        # Create histogram
        bins = np.linspace(0, max(mag_rnn.max(), mag_theory.max()) * 1.1, 30)

        ax.hist(
            mag_rnn, bins=bins, alpha=0.6, label="RNN", color="C0", edgecolor="black"
        )
        ax.hist(
            mag_theory,
            bins=bins,
            alpha=0.6,
            label="Theory",
            color="C1",
            edgecolor="black",
        )

        # Add vertical line at |λ|=1 (stability boundary)
        ax.axvline(
            1.0,
            color="red",
            linestyle="--",
            linewidth=2,
            label=r"$|\lambda|=1$",
            alpha=0.7,
        )

        ax.set_xlabel(r"Eigenvalue Magnitude $|\lambda|$")
        ax.set_ylabel("Count")
        ax.set_title(f"Eigenvalue Magnitude Distribution\n(seq_len={seq_len})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Distribution of Eigenvalue Magnitudes (Radius)", fontsize=14)
    fig.tight_layout()
    fig.savefig(osp.join(output_dir, "eigenvalue_magnitude_distribution.png"), dpi=150)
    plt.close()

    print(
        f"Saved eigenvalue magnitude distribution to {output_dir}/eigenvalue_magnitude_distribution.png"
    )


def plot_eigenvalue_distribution_vs_sequence_length(
    all_results, seq_lengths, output_dir
):
    """
    Plot the distribution of eigenvalue magnitudes with sequence length on the x-axis.
    Shows how eigenvalue spectrum changes across different sequence lengths.
    """
    print("\n=== Plotting Eigenvalue Distribution vs Sequence Length ===")

    # Collect eigenvalue magnitudes for each sequence length
    rnn_mags_by_length = []
    theory_mags_by_length = []

    for seq_len in seq_lengths:
        pca_results = all_results[seq_len]["pca_results"]
        evals_rnn = pca_results["evals_rnn"]
        evals_theory = pca_results["evals_theory"]

        rnn_mags_by_length.append(np.abs(evals_rnn))
        theory_mags_by_length.append(np.abs(evals_theory))

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Violin plot of eigenvalue magnitudes vs sequence length
    ax = axes[0]

    # Prepare data for violin plots
    positions = np.arange(len(seq_lengths))

    # RNN eigenvalues
    parts_rnn = ax.violinplot(
        rnn_mags_by_length,
        positions=positions - 0.2,
        widths=0.35,
        showmeans=True,
        showmedians=True,
    )
    for pc in parts_rnn["bodies"]:
        pc.set_facecolor("C0")
        pc.set_alpha(0.6)

    # Theory eigenvalues
    parts_theory = ax.violinplot(
        theory_mags_by_length,
        positions=positions + 0.2,
        widths=0.35,
        showmeans=True,
        showmedians=True,
    )
    for pc in parts_theory["bodies"]:
        pc.set_facecolor("C1")
        pc.set_alpha(0.6)

    ax.axhline(
        1.0, color="red", linestyle="--", linewidth=2, label=r"$|\lambda|=1$", alpha=0.7
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(seq_lengths)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Eigenvalue Magnitude |λ|")
    ax.set_title("Distribution of Eigenvalue Magnitudes vs Sequence Length")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(
        [
            plt.Line2D([0], [0], color="C0", linewidth=3),
            plt.Line2D([0], [0], color="C1", linewidth=3),
            plt.Line2D([0], [0], color="red", linestyle="--", linewidth=2),
        ],
        ["RNN", "Theory", "|λ|=1"],
    )

    # Plot 2: Box plot with scatter overlay
    ax = axes[1]

    # Create scatter plot with jitter
    for i, seq_len in enumerate(seq_lengths):
        # RNN eigenvalues
        rnn_mags = rnn_mags_by_length[i]
        x_rnn = np.random.normal(i - 0.2, 0.04, size=len(rnn_mags))
        ax.scatter(
            x_rnn, rnn_mags, alpha=0.4, s=20, color="C0", label="RNN" if i == 0 else ""
        )

        # Theory eigenvalues
        theory_mags = theory_mags_by_length[i]
        x_theory = np.random.normal(i + 0.2, 0.04, size=len(theory_mags))
        ax.scatter(
            x_theory,
            theory_mags,
            alpha=0.4,
            s=20,
            color="C1",
            label="Theory" if i == 0 else "",
        )

    # Add box plots
    bp_rnn = ax.boxplot(
        rnn_mags_by_length,
        positions=positions - 0.2,
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="C0", alpha=0.3),
        medianprops=dict(color="C0", linewidth=2),
    )

    bp_theory = ax.boxplot(
        theory_mags_by_length,
        positions=positions + 0.2,
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="C1", alpha=0.3),
        medianprops=dict(color="C1", linewidth=2),
    )

    ax.axhline(
        1.0, color="red", linestyle="--", linewidth=2, label=r"$|\lambda|=1$", alpha=0.7
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(seq_lengths)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Eigenvalue Magnitude |λ|")
    ax.set_title("Eigenvalue Magnitudes by Sequence Length (with individual values)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    fig.tight_layout()
    fig.savefig(osp.join(output_dir, "eigenvalue_distribution_vs_length.png"), dpi=150)
    plt.close()

    print(
        f"Saved eigenvalue distribution vs length to {output_dir}/eigenvalue_distribution_vs_length.png"
    )

    # Also create a summary statistics plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Collect statistics
    rnn_means = [np.mean(mags) for mags in rnn_mags_by_length]
    rnn_stds = [np.std(mags) for mags in rnn_mags_by_length]
    rnn_maxs = [np.max(mags) for mags in rnn_mags_by_length]
    rnn_above_one = [np.sum(mags > 1.0) for mags in rnn_mags_by_length]

    theory_means = [np.mean(mags) for mags in theory_mags_by_length]
    theory_stds = [np.std(mags) for mags in theory_mags_by_length]
    theory_maxs = [np.max(mags) for mags in theory_mags_by_length]
    theory_above_one = [np.sum(mags > 1.0) for mags in theory_mags_by_length]

    # Plot mean magnitudes
    ax = axes[0, 0]
    ax.plot(
        seq_lengths, rnn_means, "o-", label="RNN", linewidth=2, markersize=8, color="C0"
    )
    ax.plot(
        seq_lengths,
        theory_means,
        "s-",
        label="Theory",
        linewidth=2,
        markersize=8,
        color="C1",
    )
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Mean |λ|")
    ax.set_title("Mean Eigenvalue Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot max magnitudes
    ax = axes[0, 1]
    ax.plot(
        seq_lengths, rnn_maxs, "o-", label="RNN", linewidth=2, markersize=8, color="C0"
    )
    ax.plot(
        seq_lengths,
        theory_maxs,
        "s-",
        label="Theory",
        linewidth=2,
        markersize=8,
        color="C1",
    )
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Max |λ|")
    ax.set_title("Maximum Eigenvalue Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot standard deviation
    ax = axes[1, 0]
    ax.plot(
        seq_lengths, rnn_stds, "o-", label="RNN", linewidth=2, markersize=8, color="C0"
    )
    ax.plot(
        seq_lengths,
        theory_stds,
        "s-",
        label="Theory",
        linewidth=2,
        markersize=8,
        color="C1",
    )
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Std |λ|")
    ax.set_title("Standard Deviation of Eigenvalue Magnitudes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot count of eigenvalues with |λ| > 1
    ax = axes[1, 1]
    ax.plot(
        seq_lengths,
        rnn_above_one,
        "o-",
        label="RNN",
        linewidth=2,
        markersize=8,
        color="C0",
    )
    ax.plot(
        seq_lengths,
        theory_above_one,
        "s-",
        label="Theory",
        linewidth=2,
        markersize=8,
        color="C1",
    )
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Count")
    ax.set_title("Number of Eigenvalues with |λ| > 1")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Eigenvalue Statistics vs Sequence Length", fontsize=14)
    fig.tight_layout()
    fig.savefig(osp.join(output_dir, "eigenvalue_statistics_vs_length.png"), dpi=150)
    plt.close()

    print(
        f"Saved eigenvalue statistics vs length to {output_dir}/eigenvalue_statistics_vs_length.png"
    )


def analyze_subspace_overlap(all_results, seq_lengths, emt_linear_model, output_dir):
    """
    Analyze how different sequence length subspaces overlap in the learned W_hh.

    This reveals whether the network learns separate or overlapping representations
    for different sequence lengths.

    Args:
        all_results: Dictionary of results for each sequence length
        seq_lengths: List of sequence lengths analyzed
        emt_linear_model: EMT linear model with W_hh
        output_dir: Output directory for plots
    """
    print("\n=== Analyzing Subspace Overlap ===")

    n_lengths = len(seq_lengths)
    W_hh = emt_linear_model.W_hh

    # Extract PCA bases for each sequence length
    subspace_bases = {}
    task_dims = {}

    for seq_len in seq_lengths:
        pca_results = all_results[seq_len]["pca_results"]
        task_dim = all_results[seq_len]["task_dim"]

        # Get top principal components for this sequence length
        U_dominant = pca_results["U_dominant"]

        subspace_bases[seq_len] = U_dominant
        task_dims[seq_len] = task_dim

        print(f"Seq length {seq_len}: {task_dim}D subspace")

    # Compute overlap matrix (cosine similarity between subspaces)
    # Using principal angles / subspace overlap metric
    overlap_matrix = np.zeros((n_lengths, n_lengths))

    for i, len_i in enumerate(seq_lengths):
        for j, len_j in enumerate(seq_lengths):
            if i == j:
                overlap_matrix[i, j] = 1.0
            else:
                # Compute overlap via Frobenius norm of projection
                basis_i = subspace_bases[len_i]
                basis_j = subspace_bases[len_j]

                # Overlap = ||U_i^T @ U_j||_F^2 / (dim_i * dim_j)
                # Normalized to [0, 1]
                projection = basis_i.T @ basis_j
                overlap = np.linalg.norm(projection, "fro") ** 2
                overlap /= min(task_dims[len_i], task_dims[len_j])

                overlap_matrix[i, j] = overlap

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Overlap heatmap
    ax = axes[0]
    im = ax.imshow(overlap_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_lengths))
    ax.set_yticks(range(n_lengths))
    ax.set_xticklabels([f"Len {l}" for l in seq_lengths])
    ax.set_yticklabels([f"Len {l}" for l in seq_lengths])
    ax.set_title(
        "Subspace Overlap Matrix\n(Normalized Projection)",
        fontsize=11,
        fontweight="bold",
    )

    # Add text annotations
    for i in range(n_lengths):
        for j in range(n_lengths):
            ax.text(
                j,
                i,
                f"{overlap_matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    plt.colorbar(im, ax=ax, label="Overlap (0=orthogonal, 1=identical)")

    # Panel 2: Variance captured by each subspace
    ax = axes[1]
    variance_ratios = [
        all_results[l]["pca_results"]["variance_ratio"] for l in seq_lengths
    ]
    task_dims_list = [task_dims[l] for l in seq_lengths]

    x_pos = np.arange(n_lengths)
    bars = ax.bar(x_pos, variance_ratios, alpha=0.7, edgecolor="black")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"Len {l}\n({d}D)" for l, d in zip(seq_lengths, task_dims_list)]
    )
    ax.set_ylabel("Variance Captured", fontsize=10)
    ax.set_title(
        "Variance Captured by Task-Specific Subspaces", fontsize=11, fontweight="bold"
    )
    ax.set_ylim([0, 1])
    ax.axhline(0.9, color="r", linestyle="--", linewidth=1, alpha=0.5, label="90%")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    # Add percentage labels on bars
    for i, (bar, val) in enumerate(zip(bars, variance_ratios)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Panel 3: Cumulative explained variance across task dimensions
    ax = axes[2]

    # Perform SVD on W_hh to get singular values
    U, S, Vh = np.linalg.svd(W_hh, full_matrices=False)
    cumsum_variance = np.cumsum(S**2) / np.sum(S**2)

    ax.plot(cumsum_variance, linewidth=2, color="C0", label="All dimensions")

    # Mark task-specific dimensions
    colors = plt.cm.tab10(np.linspace(0, 1, n_lengths))
    for idx, seq_len in enumerate(seq_lengths):
        task_dim = task_dims[seq_len]
        var_at_dim = cumsum_variance[task_dim - 1]
        ax.axvline(
            task_dim,
            color=colors[idx],
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Len {seq_len} ({task_dim}D: {var_at_dim * 100:.1f}%)",
        )

    ax.set_xlabel("Number of Dimensions", fontsize=10)
    ax.set_ylabel("Cumulative Variance Explained", fontsize=10)
    ax.set_title(
        r"Cumulative Variance \& Task Dimensions", fontsize=11, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlim([0, max(task_dims_list) * 1.5])
    ax.set_ylim([0, 1])

    fig.tight_layout()
    fig.savefig(osp.join(output_dir, "subspace_overlap_analysis.png"), dpi=150)
    plt.close()

    print(
        f"Saved subspace overlap analysis to {output_dir}/subspace_overlap_analysis.png"
    )

    # Print summary statistics
    print("\nSubspace Overlap Summary:")
    for i, len_i in enumerate(seq_lengths):
        for j, len_j in enumerate(seq_lengths):
            if i < j:  # Only print upper triangle
                print(f"  Len {len_i} <-> Len {len_j}: {overlap_matrix[i, j]:.3f}")

    print("\nVariance Captured:")
    for seq_len, var_ratio in zip(seq_lengths, variance_ratios):
        print(f"  Len {seq_len}: {var_ratio * 100:.1f}%")

    return overlap_matrix


def evaluate_single_model(
    model_dir,
    model_name,
    accelerator="gpu",
    test_horizon=200,
    seq_lengths_override=None,
):
    """
    Evaluate a single model and save results to its respective directory.

    Args:
        model_dir: Path to model directory (e.g., "50_models_seqlen_8/model-38")
        model_name: Model identifier (e.g., "model-38")
        accelerator: "gpu" or "cpu"
        test_horizon: Number of timesteps for evaluation
        seq_lengths_override: Optional list of sequence lengths to analyze (overrides hparams)

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'=' * 80}")
    print(f"EVALUATING: {model_name}")
    print(f"{'=' * 80}")

    # Find checkpoint file
    ckpt_pattern = osp.join(
        model_dir, "lightning_logs", "version_*", "checkpoints", "*.ckpt"
    )
    ckpt_files = glob.glob(ckpt_pattern)

    if not ckpt_files:
        print(f"ERROR: No checkpoint found for {model_name} at {ckpt_pattern}")
        return None

    model_path = ckpt_files[0]  # Take the first (should be only one)
    print(f"Found checkpoint: {model_path}")

    # Create output directory
    output_dir = osp.join(model_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if accelerator == "gpu" and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # 1. Load model
        lmodel, emt_linear_model, hparams, batch_size = load_model(model_path, device)

        # 2. Evaluate on test data
        y_true, y_pred, seq_length, accuracy, is_eos, input_data = evaluate_model(
            lmodel, batch_size, test_horizon, device
        )

        # 3. Construct theoretical solution for the sampled sequence length
        phi_small, phi_full, task_dim = construct_theoretical_solution(
            hparams,
            seq_length,
            lmodel.input_dim,
            emt_linear_model.W_hh.shape[0],
            lmodel,
        )

        # 4. Analyze ALL sequence lengths the model was trained on
        all_results, seq_lengths = analyze_all_sequence_lengths(
            lmodel,
            emt_linear_model,
            hparams,
            output_dir,
            seq_lengths_override=seq_lengths_override,
        )

        # 5. Generate visualizations
        plot_predictions(
            y_true, y_pred, output_dir, is_eos=is_eos, input_data=input_data
        )
        plot_theoretical_matrices(all_results, seq_lengths, output_dir)
        plot_eigenspectrum_full(emt_linear_model.W_hh, phi_full, output_dir)

        # 6. PCA-based subspace analysis for sampled sequence
        pca_results = task_specific_subspace(emt_linear_model, task_dim, phi_small)
        plot_pca_analysis(pca_results, task_dim, output_dir)

        # 7. Multi-length comparison visualization
        plot_multi_length_comparison(all_results, seq_lengths, output_dir)

        # 7b. Subspace overlap analysis (only if multiple sequence lengths)
        if len(seq_lengths) > 1:
            analyze_subspace_overlap(
                all_results, seq_lengths, emt_linear_model, output_dir
            )

        # 8. Empirical Phi for all sequence lengths
        empirical_phis, f_operators = plot_empirical_phi_all_lengths(
            lmodel,
            emt_linear_model,
            hparams,
            seq_lengths,
            batch_size,
            device,
            output_dir,
        )

        # 8b. Variable basis eigenspectra (using empirical Phi)
        plot_variable_basis_eigenspectra(
            empirical_phis, f_operators, seq_lengths, lmodel, output_dir
        )

        # 9. Eigenvalue angle distribution (theta on x-axis)
        plot_eigenvalue_angle_distribution(all_results, seq_lengths, output_dir)

        # 10. Eigenvalue magnitude distribution (radius on x-axis)
        plot_eigenvalue_magnitude_distribution(all_results, seq_lengths, output_dir)

        # 11. Eigenvalue distribution vs sequence length (length on x-axis)
        plot_eigenvalue_distribution_vs_sequence_length(
            all_results, seq_lengths, output_dir
        )

        print(f"\n{'=' * 80}")
        print(f"COMPLETED: {model_name}")
        print(f"Results saved to: {output_dir}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"{'=' * 80}\n")

        # Collect metrics for this model
        metrics = {
            "model_name": model_name,
            "model_path": model_path,
            "output_dir": output_dir,
            "accuracy": float(accuracy),
            "seq_lengths": seq_lengths,
            "hidden_dim": int(hparams.get("hidden_dim", 128)),
            "input_dim": int(hparams.get("input_dim", 8)),
            "batch_size": int(batch_size),
        }

        # Add PCA metrics for each sequence length
        for seq_len in seq_lengths:
            pca_res = all_results[seq_len]["pca_results"]
            metrics[f"variance_ratio_seq{seq_len}"] = float(pca_res["variance_ratio"])
            metrics[f"frob_error_seq{seq_len}"] = float(pca_res["frob_error"])
            metrics[f"frob_relative_seq{seq_len}"] = float(pca_res["frob_relative"])

            spectral_err = pca_res.get("spectral_error", np.nan)
            if not np.isnan(spectral_err):
                metrics[f"spectral_error_deg_seq{seq_len}"] = float(
                    np.degrees(spectral_err)
                )

        # Create eigenvalue results for global analysis
        eigenvalue_results = {}
        for seq_len in seq_lengths:
            pca_res = all_results[seq_len]["pca_results"]
            eigenvalue_results[seq_len] = {
                "evals_rnn": pca_res["evals_rnn"],
                "evals_theory": pca_res["evals_theory"],
            }

        # Store raw eigenvalues directly (permutation-invariant representation)
        # Histograms can be computed later if needed, but they impose artificial ordering

        for seq_len in seq_lengths:
            pca_res = all_results[seq_len]["pca_results"]

            # Store RNN eigenvalues as complex numbers
            evals_rnn = pca_res["evals_rnn"]
            # Convert complex array to list of [real, imag] pairs for JSON serialization
            metrics[f"eigenvalues_rnn_seq{seq_len}"] = [
                [float(e.real), float(e.imag)] for e in evals_rnn
            ]

            # Store theory eigenvalues as complex numbers
            evals_theory = pca_res["evals_theory"]
            metrics[f"eigenvalues_theory_seq{seq_len}"] = [
                [float(e.real), float(e.imag)] for e in evals_theory
            ]

        return metrics, eigenvalue_results

    except Exception as e:
        print(f"\nERROR evaluating {model_name}: {e}")
        import traceback

        traceback.print_exc()
        return None


def plot_global_eigenvalue_distributions(
    models_base_dir, seq_lengths, all_model_results, magnitude_threshold=0.0
):
    """
    Create global plots aggregating eigenvalue distributions across ALL models.

    Args:
        models_base_dir: Base directory containing all models
        seq_lengths: List of sequence lengths analyzed
        all_model_results: List of dictionaries, each containing results from one model
        magnitude_threshold: Minimum magnitude to plot eigenvalues in complex plane (default: 0.0)
    """
    print(f"\n{'=' * 80}")
    print("CREATING GLOBAL EIGENVALUE DISTRIBUTIONS ACROSS ALL MODELS")
    print(f"{'=' * 80}")

    output_dir = osp.join(models_base_dir, "global_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Collect eigenvalues from all models
    all_rnn_evals = {seq_len: [] for seq_len in seq_lengths}
    all_theory_evals = {seq_len: [] for seq_len in seq_lengths}

    for model_result in all_model_results:
        for seq_len in seq_lengths:
            if seq_len in model_result:
                all_rnn_evals[seq_len].append(model_result[seq_len]["evals_rnn"])
                all_theory_evals[seq_len].append(model_result[seq_len]["evals_theory"])

    # Flatten eigenvalues across all models for each sequence length
    rnn_evals_flat = {
        seq_len: np.concatenate(all_rnn_evals[seq_len])
        for seq_len in seq_lengths
        if len(all_rnn_evals[seq_len]) > 0
    }
    theory_evals_flat = {
        seq_len: np.concatenate(all_theory_evals[seq_len])
        for seq_len in seq_lengths
        if len(all_theory_evals[seq_len]) > 0
    }

    n_models = len(all_model_results)

    # ========== Plot 1: Global Magnitude Distribution ==========
    fig, axes = plt.subplots(1, len(seq_lengths), figsize=(6 * len(seq_lengths), 5))
    if len(seq_lengths) == 1:
        axes = [axes]

    for idx, seq_len in enumerate(seq_lengths):
        if seq_len not in rnn_evals_flat:
            continue

        ax = axes[idx]

        # Compute magnitudes
        mag_rnn = np.abs(rnn_evals_flat[seq_len])
        mag_theory = np.abs(theory_evals_flat[seq_len])

        # Create histogram
        bins = np.linspace(0, max(mag_rnn.max(), mag_theory.max()) * 1.1, 50)

        ax.hist(
            mag_rnn,
            bins=bins,
            alpha=0.6,
            label=f"RNN (n={n_models})",
            color="C0",
            edgecolor="black",
            density=True,
        )
        ax.hist(
            mag_theory,
            bins=bins,
            alpha=0.6,
            label=f"Theory (n={n_models})",
            color="C1",
            edgecolor="black",
            density=True,
        )

        # Add vertical line at |λ|=1 (stability boundary)
        ax.axvline(
            1.0,
            color="red",
            linestyle="--",
            linewidth=2,
            label=r"$|\lambda|=1$",
            alpha=0.7,
        )

        ax.set_xlabel(r"Eigenvalue Magnitude $|\lambda|$")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Global Magnitude Distribution\n(seq_len={seq_len}, {n_models} models)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Eigenvalue Magnitude Distribution Across All {n_models} Models", fontsize=14
    )
    fig.tight_layout()
    fig.savefig(osp.join(output_dir, "global_magnitude_distribution.png"), dpi=150)
    plt.close()
    print(
        f"Saved global magnitude distribution to {output_dir}/global_magnitude_distribution.png"
    )

    # ========== Plot 2: Global Angle Distribution ==========
    fig, axes = plt.subplots(1, len(seq_lengths), figsize=(6 * len(seq_lengths), 5))
    if len(seq_lengths) == 1:
        axes = [axes]

    for idx, seq_len in enumerate(seq_lengths):
        if seq_len not in rnn_evals_flat:
            continue

        ax = axes[idx]

        # Compute angles
        angle_rnn = np.angle(rnn_evals_flat[seq_len])
        angle_theory = np.angle(theory_evals_flat[seq_len])

        # Create histogram
        bins = np.linspace(-np.pi, np.pi, 50)

        ax.hist(
            angle_rnn,
            bins=bins,
            alpha=0.6,
            label=f"RNN (n={n_models})",
            color="C0",
            edgecolor="black",
            density=True,
        )
        ax.hist(
            angle_theory,
            bins=bins,
            alpha=0.6,
            label=f"Theory (n={n_models})",
            color="C1",
            edgecolor="black",
            density=True,
        )

        # Add vertical lines at key angles
        ax.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.axvline(np.pi / 2, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.axvline(-np.pi / 2, color="gray", linestyle=":", linewidth=1, alpha=0.5)

        ax.set_xlabel("Eigenvalue Angle θ (radians)")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Global Angle Distribution\n(seq_len={seq_len}, {n_models} models)"
        )
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Eigenvalue Angle Distribution Across All {n_models} Models", fontsize=14
    )
    fig.tight_layout()
    fig.savefig(osp.join(output_dir, "global_angle_distribution.png"), dpi=150)
    plt.close()
    print(
        f"Saved global angle distribution to {output_dir}/global_angle_distribution.png"
    )

    # ========== Plot 3: Global Complex Plane Scatter ==========
    fig, axes = plt.subplots(1, len(seq_lengths), figsize=(7 * len(seq_lengths), 7))
    if len(seq_lengths) == 1:
        axes = [axes]

    for idx, seq_len in enumerate(seq_lengths):
        if seq_len not in rnn_evals_flat:
            continue

        ax = axes[idx]

        rnn_evals = rnn_evals_flat[seq_len]
        theory_evals = theory_evals_flat[seq_len]

        # Filter eigenvalues by magnitude threshold
        rnn_mask = np.abs(rnn_evals) >= magnitude_threshold
        theory_mask = np.abs(theory_evals) >= magnitude_threshold

        rnn_evals_filtered = rnn_evals[rnn_mask]
        theory_evals_filtered = theory_evals[theory_mask]

        # Scatter plot in complex plane
        ax.scatter(
            rnn_evals_filtered.real,
            rnn_evals_filtered.imag,
            alpha=0.3,
            s=10,
            color="C0",
            label=f"RNN (n={len(rnn_evals_filtered)})",
        )
        ax.scatter(
            theory_evals_filtered.real,
            theory_evals_filtered.imag,
            alpha=0.3,
            s=10,
            color="C1",
            label=f"Theory (n={len(theory_evals_filtered)})",
        )

        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        title = f"Global Eigenvalue Distribution\n(seq_len={seq_len}, {n_models} models"
        if magnitude_threshold > 0:
            title += f", |λ|≥{magnitude_threshold:.2f}"
        title += ")"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)

        # Add unit circle
        unit_circle = patches.Circle(
            (0.0, 0.0),
            1.0,
            fill=False,
            color="red",
            linestyle="--",
            linewidth=2,
            label=r"$|\lambda|=1$",
        )
        ax.add_patch(unit_circle)
        ax.set_aspect("equal")
        ax.legend()

    title_suffix = (
        f" (|λ|≥{magnitude_threshold:.2f})" if magnitude_threshold > 0 else ""
    )
    fig.suptitle(
        f"Eigenvalue Complex Plane Distribution Across All {n_models} Models{title_suffix}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(osp.join(output_dir, "global_complex_plane.png"), dpi=150)
    plt.close()
    print(f"Saved global complex plane plot to {output_dir}/global_complex_plane.png")
    if magnitude_threshold > 0:
        print(
            f"  (filtered to show only eigenvalues with |λ| >= {magnitude_threshold:.2f})"
        )

    # ========== Plot 4: Global Statistics Summary ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Collect statistics per model
    model_mean_mags = {seq_len: [] for seq_len in seq_lengths}
    model_max_mags = {seq_len: [] for seq_len in seq_lengths}
    model_spectral_radii = {seq_len: [] for seq_len in seq_lengths}
    model_above_one_counts = {seq_len: [] for seq_len in seq_lengths}

    for model_result in all_model_results:
        for seq_len in seq_lengths:
            if seq_len in model_result:
                evals = model_result[seq_len]["evals_rnn"]
                mags = np.abs(evals)
                model_mean_mags[seq_len].append(np.mean(mags))
                model_max_mags[seq_len].append(np.max(mags))
                model_spectral_radii[seq_len].append(np.max(mags))
                model_above_one_counts[seq_len].append(np.sum(mags > 1.0))

    # Plot 1: Distribution of mean magnitudes across models
    ax = axes[0, 0]
    for seq_len in seq_lengths:
        if len(model_mean_mags[seq_len]) > 0:
            ax.hist(
                model_mean_mags[seq_len],
                bins=20,
                alpha=0.6,
                label=f"seq_len={seq_len}",
                edgecolor="black",
            )
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel(r"Mean Eigenvalue Magnitude $|\lambda|$")
    ax.set_ylabel("Number of Models")
    ax.set_title("Distribution of Mean Eigenvalue Magnitudes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Distribution of spectral radii across models
    ax = axes[0, 1]
    for seq_len in seq_lengths:
        if len(model_spectral_radii[seq_len]) > 0:
            ax.hist(
                model_spectral_radii[seq_len],
                bins=20,
                alpha=0.6,
                label=f"seq_len={seq_len}",
                edgecolor="black",
            )
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Spectral Radius (max |λ|)")
    ax.set_ylabel("Number of Models")
    ax.set_title("Distribution of Spectral Radii")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Distribution of unstable eigenvalue counts
    ax = axes[1, 0]
    for seq_len in seq_lengths:
        if len(model_above_one_counts[seq_len]) > 0:
            ax.hist(
                model_above_one_counts[seq_len],
                bins=range(0, int(max(model_above_one_counts[seq_len])) + 2),
                alpha=0.6,
                label=f"seq_len={seq_len}",
                edgecolor="black",
            )
    ax.set_xlabel("Number of Eigenvalues with |λ| > 1")
    ax.set_ylabel("Number of Models")
    ax.set_title("Distribution of Unstable Eigenvalue Counts")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Box plot of statistics across sequence lengths
    ax = axes[1, 1]
    positions = np.arange(len(seq_lengths))
    box_data = [model_spectral_radii[seq_len] for seq_len in seq_lengths]
    ax.boxplot(
        box_data,
        positions=positions,
        patch_artist=True,
        boxprops=dict(facecolor="C0", alpha=0.6),
        medianprops=dict(color="red", linewidth=2),
    )
    ax.axhline(
        1.0, color="red", linestyle="--", linewidth=1, alpha=0.5, label=r"$|\lambda|=1$"
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{sl}" for sl in seq_lengths])
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Spectral Radius")
    ax.set_title("Spectral Radius Distribution by Sequence Length")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    fig.suptitle(f"Global Statistics Across All {n_models} Models", fontsize=14)
    fig.tight_layout()
    fig.savefig(osp.join(output_dir, "global_statistics_summary.png"), dpi=150)
    plt.close()
    print(
        f"Saved global statistics summary to {output_dir}/global_statistics_summary.png"
    )

    print(f"\nGlobal analysis saved to: {output_dir}")
    print(f"{'=' * 80}\n")


def load_config(models_dir):
    """Load training configuration from models directory if it exists."""
    config_path = osp.join(models_dir, "config.json")
    if osp.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return None


def save_eval_config(config, models_dir):
    """Save evaluation configuration to models directory."""
    eval_config_path = osp.join(models_dir, "eval_config.json")
    with open(eval_config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Evaluation configuration saved to: {eval_config_path}")
    return eval_config_path


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate multiple RNN models trained on repeat copy task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load configuration from training and evaluate (recommended)
  python evaluate_multiple.py --models_dir my_experiment

  # Override sequence lengths from training config
  python evaluate_multiple.py --models_dir my_experiment --seq_length 8

  # Variable sequence lengths (e.g., models trained with seqlen ranging from 3 to 7)
  python evaluate_multiple.py --models_dir my_models --seq_length 3 7

  # Evaluate on CPU instead of GPU
  python evaluate_multiple.py --models_dir my_experiment --accelerator cpu
        """,
    )

    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="Directory containing model subdirectories and config.json",
    )

    parser.add_argument(
        "--seq_length",
        type=int,
        nargs="+",
        help="Sequence length(s) to analyze. Single value for fixed length (e.g., 8), "
        "or two values for range (e.g., 3 7 for lengths 3-7). "
        "If not specified, will be loaded from training config.",
    )

    parser.add_argument(
        "--test_horizon",
        type=int,
        default=200,
        help="Number of timesteps for evaluation (default: 200)",
    )

    parser.add_argument(
        "--accelerator",
        type=str,
        choices=["gpu", "cpu", "auto"],
        default="auto",
        help="Device to use for evaluation (default: auto)",
    )

    parser.add_argument(
        "--model_pattern",
        type=str,
        default="model-*",
        help="Pattern to match model directories (default: model-*)",
    )

    parser.add_argument(
        "--magnitude_threshold",
        type=float,
        default=0.0,
        help="Minimum magnitude threshold for plotting eigenvalues in complex plane (default: 0.0)",
    )

    return parser.parse_args()


def main():
    """
    Main evaluation pipeline for multiple models.
    Evaluates all models in a directory and saves results to respective folders.
    """
    # Parse command-line arguments
    args = parse_args()
    models_base_dir = args.models_dir

    # Load training configuration if available
    train_config = load_config(models_base_dir)

    # Determine sequence lengths (CLI override or from training config)
    if args.seq_length is not None:
        # Use CLI argument
        if len(args.seq_length) == 1:
            seq_lengths = [args.seq_length[0]]
        elif len(args.seq_length) == 2:
            seq_lengths = list(range(args.seq_length[0], args.seq_length[1] + 1))
        else:
            raise ValueError(
                "--seq_length must have 1 value (fixed) or 2 values (range)"
            )
        print(f"Using sequence lengths from CLI: {seq_lengths}")
    elif train_config is not None:
        # Load from training config
        seq_length_param = train_config["task"]["seq_length"]
        if isinstance(seq_length_param, list):
            seq_lengths = list(range(seq_length_param[0], seq_length_param[1] + 1))
        else:
            seq_lengths = [seq_length_param]
        print(f"Using sequence lengths from training config: {seq_lengths}")
    else:
        raise ValueError(
            "No sequence length specified. Either provide --seq_length or ensure "
            "config.json exists in the models directory."
        )

    # Determine accelerator
    if args.accelerator == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    else:
        accelerator = args.accelerator

    # Create evaluation configuration
    eval_config = {
        "metadata": {
            "evaluated_at": datetime.now().isoformat(),
            "script": "evaluate_multiple.py",
            "models_dir": models_base_dir,
        },
        "evaluation": {
            "seq_lengths": seq_lengths,
            "test_horizon": args.test_horizon,
            "accelerator": accelerator,
            "model_pattern": args.model_pattern,
            "magnitude_threshold": args.magnitude_threshold,
        },
    }

    # Add training config reference if available
    if train_config is not None:
        eval_config["training_config"] = train_config

    # Save evaluation configuration
    save_eval_config(eval_config, models_base_dir)

    print(f"\n{'=' * 80}")
    print("EVALUATION CONFIGURATION")
    print(f"{'=' * 80}")
    print(f"Models directory: {models_base_dir}")
    if train_config is not None:
        print(f"Training experiment: {train_config['metadata']['experiment_name']}")
        print(f"Training date: {train_config['metadata']['created_at']}")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Test horizon: {args.test_horizon}")
    print(f"Accelerator: {accelerator}")
    print(f"Model pattern: {args.model_pattern}")
    print(f"Magnitude threshold: {args.magnitude_threshold}")
    print(f"{'=' * 80}\n")

    if not osp.exists(models_base_dir):
        print(f"ERROR: Directory {models_base_dir} not found")
        return

    # Find all model directories
    model_dirs = sorted(
        [
            d
            for d in glob.glob(osp.join(models_base_dir, args.model_pattern))
            if osp.isdir(d)
        ]
    )

    print(f"Found {len(model_dirs)} models to evaluate")
    print(f"Models: {[osp.basename(d) for d in model_dirs][:10]}")
    if len(model_dirs) > 10:
        print(f"  ... and {len(model_dirs) - 10} more")

    # Store all metrics and eigenvalue results
    all_metrics = []
    all_eigenvalue_results = []

    # Evaluate each model
    for idx, model_dir in enumerate(model_dirs):
        model_name = osp.basename(model_dir)
        print(f"\n[{idx + 1}/{len(model_dirs)}] Processing {model_name}")

        result = evaluate_single_model(
            model_dir,
            model_name,
            accelerator=accelerator,
            test_horizon=args.test_horizon,
            seq_lengths_override=seq_lengths,
        )

        if result is not None:
            metrics, eigenvalue_results = result
            all_metrics.append(metrics)
            all_eigenvalue_results.append(eigenvalue_results)

    # Save summary metrics to a JSON file
    summary_file = osp.join(models_base_dir, "evaluation_summary.json")
    with open(summary_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "n_models": len(all_metrics),
                "evaluation_config": eval_config,
                "metrics": all_metrics,
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 80}")
    print("ALL EVALUATIONS COMPLETE")
    print(f"{'=' * 80}")
    print(f"Evaluated {len(all_metrics)} / {len(model_dirs)} models successfully")
    print(f"Summary saved to: {summary_file}")
    print(f"Configuration saved to: {models_base_dir}/eval_config.json")

    # Print summary statistics
    if all_metrics:
        accuracies = [m["accuracy"] for m in all_metrics]
        print("\nAccuracy Statistics:")
        print(f"  Mean: {np.mean(accuracies):.4f}")
        print(f"  Std:  {np.std(accuracies):.4f}")
        print(f"  Min:  {np.min(accuracies):.4f}")
        print(f"  Max:  {np.max(accuracies):.4f}")

    # Create global eigenvalue distribution plots across all models
    if len(all_eigenvalue_results) > 0:
        plot_global_eigenvalue_distributions(
            models_base_dir,
            seq_lengths,
            all_eigenvalue_results,
            magnitude_threshold=args.magnitude_threshold,
        )

    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
