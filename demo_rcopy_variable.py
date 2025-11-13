import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.widgets
import mpl_toolkits.axes_grid1
import numpy as np
import pandas as pd
import torch
from emt_tools.models.linearModel import LinearModel
from emt_tools.utils import spectral_comparison
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from mpl_toolkits.axes_grid1 import make_axes_locatable

from em_discrete.models.rnn_model import RNNModel
from em_discrete.tasks.variable_linear_vb import BinaryLinearVarVBTask
from em_discrete.utils.result_handling import parse_directory

EXPERIMENT_OUTPUT_DIR = "./"
EXPERIMENT_NAME = "rcopy_test"
EXPERIMENT_PATH = os.path.join(EXPERIMENT_OUTPUT_DIR, EXPERIMENT_NAME)
TASK_ID = (
    255,
    0,
    0,
)  # this is the task id for repeat copy. Dont change unless you know what you are doing - some alternate ones are below
# TASK_ID = (9241421688590303745, 262676, 189) # \mathcal{T}_2 in the paper
# TASK_ID = (9241421688590303745, 0, 0) # \mathcal{T}_3
# TASK_ID = (9241421688590303745, 38637, 145) # \mathcal{T}_4
# we denote each task in the VB tasks by a three tuple representation
# 1st number in the tuple encodes the choice of dimensions - d dimensions chosen from s*d possibilities
# 2nd number denotes a permutation of the dimensions - d! possibilities
# 3rd number denotes the sign of the linear operator - +/- 2^d possibilities
SEED = 0
INPUT_DIM = 8  # Data dimension (number of task features)
SEQ_LENGTH = (7, 8)
HIDDEN_DIM = 128
LEARNING_RATE = (
    1e-3  # Balanced LR for warm restart scheduler (will periodically reset to this)
)
BATCH_SIZE = 64
GRADIENT_CLIP = 1.0
L2_PENALTY = 0.0
# L2_PENALTY = 0.001
CURRICULUM_THRESHOLD = 0.985  # Increased from 0.96 to ensure mastery before advancing
MAX_EPOCHS = 300  # Increased from 200 to allow more training time
SAVE_DIR = os.path.join(EXPERIMENT_OUTPUT_DIR, EXPERIMENT_NAME)
ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
USE_EOS = True  # Use EOS marker for variable-length sequences

seed_everything(SEED)

# Model dimensions: input includes EOS channel if enabled, output is always just data
MODEL_INPUT_DIM = INPUT_DIM + 1 if USE_EOS else INPUT_DIM
MODEL_OUTPUT_DIM = INPUT_DIM

print(f"Configuration:")
print(f"  Data dimension: {INPUT_DIM}")
print(f"  Use EOS: {USE_EOS}")
print(f"  Model input dimension: {MODEL_INPUT_DIM}")
print(f"  Model output dimension: {MODEL_OUTPUT_DIM}")
print(f"  Sequence length range: {SEQ_LENGTH}")
print()

model = RNNModel(
    MODEL_INPUT_DIM, HIDDEN_DIM, MODEL_OUTPUT_DIM, bias=False
)  # set bias False so that origin is a fixed point

lmodule = BinaryLinearVarVBTask(
    model,
    seed=SEED,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    input_dim=INPUT_DIM,
    seq_length=SEQ_LENGTH,
    model_type="rnn",
    hidden_dim=HIDDEN_DIM,
    task_id=TASK_ID,
    curriculum=True,  # enable curriculum training
    curriculum_threshold=CURRICULUM_THRESHOLD,
    l2_penalty=L2_PENALTY,
    use_eos=USE_EOS,  # Enable EOS marker for variable-length sequences
)

tb_logger = TensorBoardLogger(
    save_dir=SAVE_DIR,
    log_graph=True,
)

# Checkpoint callback to save best models
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(SAVE_DIR, "checkpoints"),
    filename="best-{epoch:02d}-{train_acc:.3f}",
    monitor="train_acc",
    mode="max",
    save_top_k=3,  # Keep top 3 checkpoints
    save_last=True,  # Also save the last checkpoint
    every_n_epochs=1,
)

# Learning rate monitor to track LR changes
lr_monitor = LearningRateMonitor(logging_interval="epoch")

trainer = Trainer(
    logger=tb_logger,
    callbacks=[checkpoint_callback, lr_monitor],
    gradient_clip_val=GRADIENT_CLIP,
    max_epochs=MAX_EPOCHS,
    deterministic=True,
    accelerator=ACCELERATOR,
    num_sanity_val_steps=0,
    default_root_dir=EXPERIMENT_NAME,
)

trainer.fit(lmodule)
