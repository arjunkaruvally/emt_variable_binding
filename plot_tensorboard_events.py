#!/usr/bin/env python3
"""
Script to decode TensorFlow events files and plot variables by name.

Usage:
    python plot_tensorboard_events.py <events_file_path> [--variables var1 var2 ...] [--list]

Features:
    - Automatically adapts plotting style based on data density
    - For dense data (>500 points): uses thin lines with smoothing overlay
    - For sparse data: uses markers for clear visualization

Examples:
    # List all available variables
    python plot_tensorboard_events.py rcopy_test/lightning_logs/version_9/events.out.tfevents.* --list

    # Plot specific variables
    python plot_tensorboard_events.py rcopy_test/lightning_logs/version_9/events.out.tfevents.* --variables train_loss train_acc

    # Plot all variables
    python plot_tensorboard_events.py rcopy_test/lightning_logs/version_9/events.out.tfevents.*

    # Plot without smoothing
    python plot_tensorboard_events.py rcopy_test/lightning_logs/version_9/events.out.tfevents.* --no-smooth
"""

import argparse
import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def load_tensorboard_events(event_file_path):
    """
    Load TensorBoard events file and extract all scalar summaries.

    Args:
        event_file_path: Path to the tfevents file

    Returns:
        Dictionary mapping tag names to lists of (step, value, wall_time) tuples
    """
    # Create an event accumulator
    ea = event_accumulator.EventAccumulator(
        event_file_path,
        size_guidance={
            event_accumulator.COMPRESSED_HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
            event_accumulator.AUDIO: 0,
            event_accumulator.SCALARS: 0,  # 0 means load all
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.TENSORS: 0,
        },
    )

    # Load the events
    ea.Reload()

    # Extract scalar data
    data = defaultdict(list)

    # Get all available tags
    tags = ea.Tags()

    # Process scalar values
    for tag in tags.get("scalars", []):
        events = ea.Scalars(tag)
        for event in events:
            data[tag].append(
                {"step": event.step, "value": event.value, "wall_time": event.wall_time}
            )

    # Also check for tensors (sometimes metrics are stored as tensors)
    for tag in tags.get("tensors", []):
        try:
            events = ea.Tensors(tag)
            for event in events:
                # Try to extract scalar value from tensor
                tensor_proto = event.tensor_proto
                if tensor_proto.dtype == 1:  # DT_FLOAT
                    if len(tensor_proto.float_val) > 0:
                        value = tensor_proto.float_val[0]
                        data[tag].append(
                            {
                                "step": event.step,
                                "value": value,
                                "wall_time": event.wall_time,
                            }
                        )
        except:
            pass

    return dict(data), tags


def list_variables(data):
    """Print all available variables/tags in the events file."""
    print("\nAvailable variables in the events file:")
    print("-" * 60)
    for i, tag in enumerate(sorted(data.keys()), 1):
        num_points = len(data[tag])
        print(f"{i}. {tag} ({num_points} data points)")
    print("-" * 60)


def smooth_data(values, window_size):
    """Apply moving average smoothing to data."""
    if len(values) < window_size:
        return values

    smoothed = np.convolve(values, np.ones(window_size) / window_size, mode="valid")
    return smoothed


def plot_variables(data, variable_names=None, save_path=None, smooth=True):
    """
    Plot variables from the TensorBoard events.

    Args:
        data: Dictionary of variable data
        variable_names: List of variable names to plot (None = plot all)
        save_path: Path to save the figure (None = display only)
        smooth: Whether to add smoothed line for dense data (default: True)
    """
    if variable_names is None:
        variable_names = sorted(data.keys())
    else:
        # Filter to only valid variable names
        valid_names = []
        for name in variable_names:
            if name in data:
                valid_names.append(name)
            else:
                print(f"Warning: Variable '{name}' not found in events file")
        variable_names = valid_names

    if not variable_names:
        print("No valid variables to plot")
        return

    # Determine grid layout
    n_vars = len(variable_names)
    n_cols = min(2, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_vars == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_vars > 1 else [axes]

    # Plot each variable
    for idx, var_name in enumerate(variable_names):
        ax = axes[idx]
        events = data[var_name]

        steps = np.array([e["step"] for e in events])
        values = np.array([e["value"] for e in events])

        n_points = len(steps)

        # Adapt plotting style based on number of data points
        if n_points > 500:
            # Dense data: no markers, thin lines, lower alpha
            ax.plot(
                steps, values, linewidth=0.5, alpha=0.3, color="C0", label="Raw data"
            )

            # Add smoothed line to show trend
            if smooth and n_points > 50:
                # Use adaptive window size based on number of points
                window_size = max(int(n_points * 0.02), 10)  # 2% of data or min 10
                window_size = min(window_size, 200)  # Cap at 200

                smoothed_values = smooth_data(values, window_size)
                # Adjust steps to match smoothed data length
                offset = (len(values) - len(smoothed_values)) // 2
                smoothed_steps = steps[offset : offset + len(smoothed_values)]

                ax.plot(
                    smoothed_steps,
                    smoothed_values,
                    linewidth=2,
                    alpha=0.9,
                    color="C0",
                    label=f"Smoothed (window={window_size})",
                )
                ax.legend(loc="best", fontsize=8)
        else:
            # Sparse data: keep markers
            ax.plot(steps, values, linewidth=2, marker="o", markersize=3, alpha=0.7)

        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.set_title(var_name)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_vars, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()


def print_variable_summary(data, variable_name):
    """Print summary statistics for a specific variable."""
    if variable_name not in data:
        print(f"Variable '{variable_name}' not found")
        return

    events = data[variable_name]
    values = [e["value"] for e in events]
    steps = [e["step"] for e in events]

    print(f"\nSummary for '{variable_name}':")
    print("-" * 60)
    print(f"Number of data points: {len(values)}")
    print(f"Step range: {min(steps)} to {max(steps)}")
    print(f"Value range: {min(values):.6f} to {max(values):.6f}")
    print(f"Mean: {np.mean(values):.6f}")
    print(f"Std: {np.std(values):.6f}")
    print(f"Final value: {values[-1]:.6f}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Decode and plot TensorFlow events files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "event_file", type=str, help="Path to the tfevents file (supports wildcards)"
    )
    parser.add_argument(
        "--variables", "-v", nargs="+", help="Variable names to plot (default: all)"
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available variables and exit",
    )
    parser.add_argument(
        "--summary",
        "-s",
        type=str,
        help="Print summary statistics for a specific variable",
    )
    parser.add_argument(
        "--save", type=str, help="Save plot to file instead of displaying"
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable smoothing for dense data (default: smoothing enabled)",
    )

    args = parser.parse_args()

    # Handle wildcards in path
    matching_files = glob.glob(args.event_file)
    if not matching_files:
        print(f"Error: No files found matching '{args.event_file}'")
        return

    if len(matching_files) > 1:
        print(f"Warning: Multiple files found, using: {matching_files[0]}")

    event_file = matching_files[0]
    print(f"Loading events from: {event_file}")

    # Load the events
    data, tags = load_tensorboard_events(event_file)

    if not data:
        print("No scalar data found in the events file")
        return

    # List variables if requested
    if args.list:
        list_variables(data)
        return

    # Print summary if requested
    if args.summary:
        print_variable_summary(data, args.summary)
        return

    # Plot variables
    plot_variables(data, args.variables, args.save, smooth=not args.no_smooth)


if __name__ == "__main__":
    main()
