import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from em_discrete.models.rnn_model import RNNModel
from em_discrete.tasks.infinite_horizon_vb import BinaryLinearVBInfiniteHorizonTask

EXPERIMENT_OUTPUT_DIR = "./"
EXPERIMENT_NAME = "rcopy_infinite_horizon"
EXPERIMENT_PATH = os.path.join(EXPERIMENT_OUTPUT_DIR, EXPERIMENT_NAME)

# Task configuration
TASK_ID = (255, 0, 0)  # Repeat copy task

# Model and training configuration
SEED = 0
INPUT_DIM = 8  # Data dimension (number of task features)
SEQ_LENGTH_RANGE = (5, 10)  # Variable sequence length range
OUTPUT_HORIZON = 15  # Number of composition outputs after each sequence
TOTAL_LENGTH = 300  # Total length of each training sample
HIDDEN_DIM = 128
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
GRADIENT_CLIP = 1.0
L2_PENALTY = 0.0
MAX_EPOCHS = 100
SAVE_DIR = os.path.join(EXPERIMENT_OUTPUT_DIR, EXPERIMENT_NAME)
ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"

seed_everything(SEED)

# Model dimensions: input includes EOS channel, output is data only
MODEL_INPUT_DIM = INPUT_DIM + 1  # +1 for EOS channel
MODEL_OUTPUT_DIM = INPUT_DIM

print("=" * 70)
print("Infinite Horizon Variable Binding Task Configuration")
print("=" * 70)
print(f"  Data dimension: {INPUT_DIM}")
print(f"  Model input dimension: {MODEL_INPUT_DIM} (includes EOS channel)")
print(f"  Model output dimension: {MODEL_OUTPUT_DIM}")
print(f"  Sequence length range: {SEQ_LENGTH_RANGE}")
print(f"  Output horizon per sequence: {OUTPUT_HORIZON}")
print(f"  Total sequence length: {TOTAL_LENGTH}")
print(f"  Hidden dimension: {HIDDEN_DIM}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Max epochs: {MAX_EPOCHS}")
print("=" * 70)
print()

# Create model
model = RNNModel(
    MODEL_INPUT_DIM, HIDDEN_DIM, MODEL_OUTPUT_DIM, bias=False
)  # bias=False so origin is a fixed point

# Create task
lmodule = BinaryLinearVBInfiniteHorizonTask(
    model,
    seed=SEED,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    input_dim=INPUT_DIM,
    seq_length_range=SEQ_LENGTH_RANGE,
    output_horizon=OUTPUT_HORIZON,
    total_length=TOTAL_LENGTH,
    model_type="rnn",
    task_id=TASK_ID,
    l2_penalty=L2_PENALTY,
)

# Logger
tb_logger = TensorBoardLogger(
    save_dir=SAVE_DIR,
    log_graph=True,
)

# Checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(SAVE_DIR, "checkpoints"),
    filename="best-{epoch:02d}-{train_acc:.3f}",
    monitor="train_acc",
    mode="max",
    save_top_k=3,
    save_last=True,
    every_n_epochs=1,
)

# Learning rate monitor
lr_monitor = LearningRateMonitor(logging_interval="epoch")

# Trainer
trainer = Trainer(
    logger=tb_logger,
    callbacks=[checkpoint_callback, lr_monitor],
    gradient_clip_val=GRADIENT_CLIP,
    max_epochs=MAX_EPOCHS,
    deterministic=True,
    accelerator=ACCELERATOR,
    num_sanity_val_steps=2,  # Run validation sanity checks
    check_val_every_n_epoch=1,
    default_root_dir=SAVE_DIR,
)

print("Starting training...")
print()
trainer.fit(lmodule)

print()
print("=" * 70)
print("Training complete!")
print(f"Checkpoints saved to: {os.path.join(SAVE_DIR, 'checkpoints')}")
print(f"TensorBoard logs saved to: {SAVE_DIR}")
print()
print("To view training progress, run:")
print(f"  tensorboard --logdir {SAVE_DIR}")
print("=" * 70)
