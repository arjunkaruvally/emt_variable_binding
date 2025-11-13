"""Diagnostic test to check if basic training works"""

import numpy as np
import torch
from lightning.pytorch import Trainer, seed_everything

from em_discrete.models.rnn_model import RNNModel
from em_discrete.tasks.binary_linearVB import BinaryLinearVBTask

# Test configuration - exactly as in working demo
SEED = 0
INPUT_DIM = 8
SEQ_LENGTH = 8
HIDDEN_DIM = 128
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
GRADIENT_CLIP = 1.0
CURRICULUM_THRESHOLD = 0.96
MAX_EPOCHS = 10
TASK_ID = (255, 0, 0)

print("=" * 70)
print("DIAGNOSTIC TEST - Fixed-Length Task")
print("=" * 70)
print(f"Task ID: {TASK_ID}")
print(f"Binary representation of {TASK_ID[0]}: {bin(TASK_ID[0])}")
print(f"Number of 1s (should be {INPUT_DIM}): {bin(TASK_ID[0]).count('1')}")
print()

# Clear GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("Cleared GPU cache")

seed_everything(SEED)

model = RNNModel(INPUT_DIM, HIDDEN_DIM, INPUT_DIM, bias=False)

# Check model initialization
print(f"\nModel W_hh norm: {torch.norm(model.W_hh).item():.4f}")
print(f"Model W_ih norm: {torch.norm(model.W_ih).item():.4f}")
print(f"Model W_ho norm: {torch.norm(model.W_ho).item():.4f}")

lmodule = BinaryLinearVBTask(
    model,
    seed=SEED,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    input_dim=INPUT_DIM,
    seq_length=SEQ_LENGTH,
    model_type="rnn",
    hidden_dim=HIDDEN_DIM,
    task_id=TASK_ID,
    curriculum=True,
    curriculum_threshold=CURRICULUM_THRESHOLD,
    l2_penalty=0.0,
)

print(f"\nCurriculum horizons: {lmodule.curriculum_horizons}")
print(f"Starting horizon: {lmodule.train_dataset.horizon}")

trainer = Trainer(
    gradient_clip_val=GRADIENT_CLIP,
    max_epochs=MAX_EPOCHS,
    deterministic=True,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=[0] if torch.cuda.is_available() else "auto",
    num_sanity_val_steps=0,
    enable_progress_bar=True,
)

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("Expected: Should reach ~100% accuracy by epoch 6-7")
print("=" * 70 + "\n")

trainer.fit(lmodule)

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
