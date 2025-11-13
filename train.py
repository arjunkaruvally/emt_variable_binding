import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from tqdm import tqdm

from em_discrete.models.rnn_model import RNNModel
from em_discrete.tasks.infinite_horizon_vb import BinaryLinearVBInfiniteHorizonTask
from em_discrete.tasks.variable_linear_vb import BinaryLinearVarVBTask


def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train multiple RNN models on variable binding tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train 50 models with fixed sequence length (regular mode)
  python train.py --experiment_name my_exp --seq_length 8 --hidden_dim 128 --num_models 50

  # Train with variable sequence lengths (regular mode)
  python train.py --experiment_name var_exp --seq_length 3 7 --hidden_dim 96 --num_models 20

  # Train with infinite horizon mode
  python train.py --experiment_name infinite_exp --seq_length 5 10 --hidden_dim 128 \\
      --infinite_horizon --output_horizon 15 --total_length 300 --num_models 20

  # Customize training parameters
  python train.py --experiment_name custom --seq_length 10 --hidden_dim 128 \\
      --learning_rate 0.0005 --epochs 30 --num_workers 4
        """,
    )

    # Experiment configuration
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the experiment (will create a directory with this name)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Base output directory for experiments (default: ./)",
    )

    # Task configuration
    parser.add_argument(
        "--task_id",
        type=int,
        nargs=3,
        default=[255, 0, 0],
        help="Task ID as three integers (default: 255 0 0 for repeat copy)",
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=8,
        help="Input dimension (default: 8)",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        nargs="+",
        required=True,
        help="Sequence length. Single value for fixed length (e.g., 8), "
        "or two values for range (e.g., 3 7 for lengths 3-7)",
    )
    parser.add_argument(
        "--use_eos",
        action="store_true",
        help="Use end-of-sequence token (default: False)",
    )
    parser.add_argument(
        "--infinite_horizon",
        action="store_true",
        help="Use infinite horizon mode where sequences are continuously generated until EOS (default: False)",
    )
    parser.add_argument(
        "--output_horizon",
        type=int,
        default=15,
        help="Number of outputs after each sequence in infinite horizon mode (default: 15)",
    )
    parser.add_argument(
        "--total_length",
        type=int,
        default=300,
        help="Total length of each training sample in infinite horizon mode (default: 300)",
    )

    # Model configuration
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension of RNN (default: 128)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="rnn",
        choices=["rnn"],
        help="Model type (default: rnn)",
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        help="Use bias in RNN (default: False, so origin is a fixed point)",
    )

    # Training configuration
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed (default: 0). Each model gets seed + model_index * 1000",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=1.0,
        help="Gradient clipping value (default: 1.0)",
    )
    parser.add_argument(
        "--l2_penalty",
        type=float,
        default=0.0,
        help="L2 regularization penalty (default: 0.0)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--num_models",
        type=int,
        default=50,
        help="Number of models to train (default: 50)",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        default=True,
        help="Use curriculum learning (default: True)",
    )
    parser.add_argument(
        "--no_curriculum",
        action="store_false",
        dest="curriculum",
        help="Disable curriculum learning",
    )
    parser.add_argument(
        "--curriculum_threshold",
        type=float,
        default=0.96,
        help="Curriculum learning threshold (default: 0.96)",
    )
    parser.add_argument(
        "--curriculum_horizons",
        type=int,
        nargs="+",
        default=[10, 20, 40, 60, 100],
        help="Curriculum horizons (default: 10 20 40 60 100)",
    )

    # Execution configuration
    parser.add_argument(
        "--accelerator",
        type=str,
        choices=["gpu", "cpu", "auto"],
        default="auto",
        help="Accelerator to use (default: auto)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of parallel training processes (default: 2)",
    )

    return parser.parse_args()


def create_config(args):
    """Create configuration dictionary from parsed arguments."""
    # Parse sequence length
    if args.infinite_horizon:
        # Infinite horizon mode requires a range
        if len(args.seq_length) == 1:
            # If single value given, create a small range around it
            seq_length = (max(1, args.seq_length[0] - 2), args.seq_length[0] + 2)
        elif len(args.seq_length) == 2:
            seq_length = tuple(args.seq_length)
        else:
            raise ValueError(
                "--seq_length must have 1 or 2 values for infinite horizon mode"
            )
    else:
        # Regular mode
        if len(args.seq_length) == 1:
            seq_length = args.seq_length[0]
        elif len(args.seq_length) == 2:
            seq_length = args.seq_length  # Will be stored as [min, max]
        else:
            raise ValueError(
                "--seq_length must have 1 value (fixed) or 2 values (range)"
            )

    # Determine accelerator
    if args.accelerator == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    else:
        accelerator = args.accelerator

    config = {
        "metadata": {
            "experiment_name": args.experiment_name,
            "created_at": datetime.now().isoformat(),
            "script": "train_multiple.py",
        },
        "task": {
            "task_id": tuple(args.task_id),
            "input_dim": args.input_dim,
            "seq_length": seq_length,
            "use_eos": (
                True if args.infinite_horizon else args.use_eos
            ),  # Infinite horizon always uses EOS
            "infinite_horizon": args.infinite_horizon,
            "output_horizon": args.output_horizon if args.infinite_horizon else None,
            "total_length": args.total_length if args.infinite_horizon else None,
        },
        "model": {
            "type": args.model_type,
            "hidden_dim": args.hidden_dim,
            "bias": args.bias,
        },
        "training": {
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_clip": args.gradient_clip,
            "l2_penalty": args.l2_penalty,
            "epochs": args.epochs,
            "num_models": args.num_models,
            "curriculum": args.curriculum,
            "curriculum_threshold": args.curriculum_threshold,
            "curriculum_horizons": args.curriculum_horizons,
        },
        "execution": {
            "accelerator": accelerator,
            "num_workers": args.num_workers,
        },
    }

    return config


def save_config(config, base_dir):
    """Save configuration to JSON file in base directory."""
    config_path = Path(base_dir) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")
    return config_path


def train_single_model(model_index, config, base_dir):
    """Train a single model with the given index."""
    import os as os_module

    # Extract config values
    task_config = config["task"]
    model_config = config["model"]
    train_config = config["training"]
    exec_config = config["execution"]

    # Create the model directory first
    this_run_dir = Path(base_dir) / f"model-{model_index + 1}"
    ckpt_path = this_run_dir / "checkpoint"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Create log file path
    log_file_path = this_run_dir / "training.log"

    # Save original stdout/stderr file descriptors
    old_stdout_fd = os_module.dup(sys.stdout.fileno())
    old_stderr_fd = os_module.dup(sys.stderr.fileno())

    try:
        # Open log file and redirect stdout/stderr to it
        log_fd = os_module.open(
            str(log_file_path),
            os_module.O_WRONLY | os_module.O_CREAT | os_module.O_TRUNC,
            0o644,
        )
        os_module.dup2(log_fd, sys.stdout.fileno())
        os_module.dup2(log_fd, sys.stderr.fileno())
        os_module.close(log_fd)

        seed = train_config["seed"] + model_index * 1000
        seed_everything(seed)

        # Print header to log file
        print("=" * 70)
        print(f"Training Model {model_index + 1}/{train_config['num_models']}")
        print(f"Seed: {seed}")
        print("=" * 70)
        print()

        # Determine model dimensions based on mode
        if task_config["infinite_horizon"]:
            # Infinite horizon: EOS encoded as all -1s within data dimensions
            # Input and output dimensions are the same (no separate EOS channel)
            model_input_dim = task_config["input_dim"]
            model_output_dim = task_config["input_dim"]
            print("Infinite horizon mode:")
            print(f"  Model input dim: {model_input_dim} (EOS encoded as all -1s)")
            print(f"  Model output dim: {model_output_dim}")
        else:
            # Regular mode
            model_input_dim = task_config["input_dim"]
            model_output_dim = task_config["input_dim"]

        model = RNNModel(
            model_input_dim,
            model_config["hidden_dim"],
            model_output_dim,
            bias=model_config["bias"],
        )

        if task_config["infinite_horizon"]:
            # Use infinite horizon task
            lmodule = BinaryLinearVBInfiniteHorizonTask(
                model,
                seed=seed,
                learning_rate=train_config["learning_rate"],
                batch_size=train_config["batch_size"],
                input_dim=task_config["input_dim"],
                seq_length_range=task_config["seq_length"],
                output_horizon=task_config["output_horizon"],
                total_length=task_config["total_length"],
                model_type=model_config["type"],
                l2_penalty=train_config["l2_penalty"],
                task_id=task_config["task_id"],
            )
        else:
            # Use regular variable VB task
            lmodule = BinaryLinearVarVBTask(
                model,
                seed=seed,
                learning_rate=train_config["learning_rate"],
                batch_size=train_config["batch_size"],
                input_dim=task_config["input_dim"],
                seq_length=task_config["seq_length"],
                model_type=model_config["type"],
                hidden_dim=model_config["hidden_dim"],
                task_id=task_config["task_id"],
                curriculum=train_config["curriculum"],
                curriculum_threshold=train_config["curriculum_threshold"],
                curriculum_horizons=train_config["curriculum_horizons"],
                l2_penalty=train_config["l2_penalty"],
                use_eos=task_config["use_eos"],
            )

        tb_logger = TensorBoardLogger(
            save_dir=this_run_dir,
            log_graph=True,
        )

        # If using GPU, assign to a specific device based on worker index
        devices = "auto"
        gpu_id = None
        if exec_config["accelerator"] == "gpu":
            # Distribute models across available GPUs
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                gpu_id = model_index % num_gpus
                devices = [gpu_id]
                print(f"Assigned to GPU {gpu_id}")

        trainer = Trainer(
            logger=tb_logger,
            gradient_clip_val=train_config["gradient_clip"],
            min_epochs=train_config["epochs"],
            max_epochs=train_config["epochs"],
            deterministic=True,
            accelerator=exec_config["accelerator"],
            devices=[0],
            num_sanity_val_steps=0,
            default_root_dir=str(this_run_dir),
            enable_progress_bar=True,  # Enable progress bar in log file
            enable_model_summary=True,  # Enable model summary in log file
        )

        trainer.fit(lmodule)
        trainer.save_checkpoint(ckpt_path)

        print()
        print("=" * 70)
        print(f"Training completed successfully for Model {model_index + 1}")
        print("=" * 70)

        # Restore original stdout/stderr
        os_module.dup2(old_stdout_fd, sys.stdout.fileno())
        os_module.dup2(old_stderr_fd, sys.stderr.fileno())
        os_module.close(old_stdout_fd)
        os_module.close(old_stderr_fd)

        return model_index, True, None, gpu_id

    except Exception as e:
        # Restore original stdout/stderr before returning
        try:
            os_module.dup2(old_stdout_fd, sys.stdout.fileno())
            os_module.dup2(old_stderr_fd, sys.stderr.fileno())
            os_module.close(old_stdout_fd)
            os_module.close(old_stderr_fd)
        except:
            pass

        return model_index, False, str(e), None


if __name__ == "__main__":
    # Parse arguments and create configuration
    args = parse_args()
    config = create_config(args)

    # Create base directory
    base_dir = os.path.join(args.output_dir, args.experiment_name)
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    # Save configuration
    save_config(config, base_dir)

    # Print configuration summary
    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Experiment: {config['metadata']['experiment_name']}")
    print(f"Base directory: {base_dir}")
    print("\nTask Configuration:")
    print(f"  Task ID: {config['task']['task_id']}")
    print(f"  Input dim: {config['task']['input_dim']}")
    if config["task"]["infinite_horizon"]:
        print("  Mode: Infinite Horizon (with EOS)")
        print(f"  Sequence length range: {config['task']['seq_length']}")
        print(f"  Output horizon: {config['task']['output_horizon']}")
        print(f"  Total length: {config['task']['total_length']}")
    else:
        print("  Mode: Regular")
        print(f"  Sequence length: {config['task']['seq_length']}")
        print(f"  Use EOS: {config['task']['use_eos']}")
    print("\nModel Configuration:")
    print(f"  Type: {config['model']['type']}")
    print(f"  Hidden dim: {config['model']['hidden_dim']}")
    print(f"  Bias: {config['model']['bias']}")
    print("\nTraining Configuration:")
    print(f"  Num models: {config['training']['num_models']}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Curriculum: {config['training']['curriculum']}")
    if config["training"]["curriculum"]:
        print(f"  Curriculum threshold: {config['training']['curriculum_threshold']}")
        print(f"  Curriculum horizons: {config['training']['curriculum_horizons']}")
    print("\nExecution Configuration:")
    print(f"  Accelerator: {config['execution']['accelerator']}")
    print(f"  Num workers: {config['execution']['num_workers']}")
    if config["execution"]["accelerator"] == "gpu":
        num_gpus = torch.cuda.device_count()
        print(f"  Number of GPUs available: {num_gpus}")
        if num_gpus == 0:
            print("  WARNING: GPU accelerator selected but no GPUs found!")
    print("=" * 70)
    print()

    # Train models in parallel with improved progress tracking
    completed = 0
    failed = []
    submitted_count = 0
    start_time = time.time()

    num_models = config["training"]["num_models"]
    num_workers = config["execution"]["num_workers"]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all jobs
        future_to_model = {
            executor.submit(train_single_model, i, config, base_dir): i
            for i in range(num_models)
        }
        submitted_count = len(future_to_model)

        # Create multiple progress bars
        # Bar 0: Overall progress
        overall_pbar = tqdm(
            total=num_models,
            desc="Overall",
            position=0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} models [{elapsed}<{remaining}]",
        )

        # Bar 1: Active workers status
        workers_pbar = tqdm(
            total=0, desc="", position=1, bar_format="{desc}", leave=True
        )

        # Bar 2-5: Recently completed models
        recent_pbars = []
        for i in range(4):
            pbar = tqdm(
                total=0, desc="", position=2 + i, bar_format="{desc}", leave=True
            )
            recent_pbars.append(pbar)

        # Bar 6: Statistics
        stats_pbar = tqdm(total=0, desc="", position=6, bar_format="{desc}", leave=True)

        recent_completions = []  # Track last few completions

        # Process completed jobs
        for future in as_completed(future_to_model):
            model_idx, success, error, gpu_id = future.result()
            completed += 1

            if not success:
                failed.append((model_idx, error))
                status_icon = "✗"
                status_text = "FAILED"
            else:
                status_icon = "✓"
                status_text = "DONE"

            # Update recent completions
            gpu_str = f"GPU{gpu_id}" if gpu_id is not None else "CPU"
            completion_msg = (
                f"{status_icon} Model {model_idx + 1:2d} [{gpu_str}] - {status_text}"
            )
            recent_completions.append(completion_msg)
            if len(recent_completions) > 4:
                recent_completions.pop(0)

            # Update recent completion bars
            for i, pbar in enumerate(recent_pbars):
                if i < len(recent_completions):
                    pbar.set_description_str(recent_completions[i])
                else:
                    pbar.set_description_str("")

            # Update overall progress
            overall_pbar.update(1)

            # Calculate statistics
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta_seconds = (num_models - completed) / rate if rate > 0 else 0
            eta = str(timedelta(seconds=int(eta_seconds)))

            # Update workers status
            active = submitted_count - completed
            workers_pbar.set_description_str(
                f"Active: {active}/{num_workers} workers | "
                f"Queued: {num_models - submitted_count}"
            )

            # Update statistics
            stats_pbar.set_description_str(
                f"✓ Success: {completed - len(failed)} | "
                f"✗ Failed: {len(failed)} | "
                f"Rate: {rate * 60:.1f} models/min | "
                f"ETA: {eta}"
            )

        # Close all progress bars
        overall_pbar.close()
        workers_pbar.close()
        for pbar in recent_pbars:
            pbar.close()
        stats_pbar.close()

    print()
    print("=" * 70)
    print(
        f"Training complete! {completed - len(failed)}/{num_models} models trained successfully"
    )

    total_time = time.time() - start_time
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    print(f"Average time per model: {total_time / completed:.1f}s")

    if failed:
        print(f"\nFailed models ({len(failed)}):")
        for model_idx, error in failed:
            print(f"  Model {model_idx + 1}: {error}")

    print(f"\nResults saved to: {base_dir}")
    print(f"Configuration: {base_dir}/config.json")
    print(f"Training logs: {base_dir}/model-N/training.log")
    print(f"Checkpoints: {base_dir}/model-N/checkpoint")
    print(f"TensorBoard logs: {base_dir}/model-N/lightning_logs")
    print("=" * 70)
