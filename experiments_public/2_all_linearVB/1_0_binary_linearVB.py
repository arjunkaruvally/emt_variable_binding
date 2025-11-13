import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decouple import config  # obtain environment files
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

# from test_tube import Experiment
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

from em_discrete.models.lstm_model import LSTMModel

# from rnn_model import *
from em_discrete.models.rnn_model import RNNModel
from em_discrete.tasks.binary_linearVB import *

parser = argparse.ArgumentParser(description="Binary Logic Task with RNNs")
parser.add_argument("--seed", type=int, default=0, help="seed")
parser.add_argument("--input_dim", type=int, default=8, help="input dimensions")
parser.add_argument(
    "--task_id",
    type=int,
    nargs="+",
    default=(255, 0, 0),
    help="Task ID for the binaryLinearVB task (task_id1, task_id2, task_id3)",
)
parser.add_argument(
    "--seq_length", type=int, default=10, help="sequence length for the memory"
)
parser.add_argument("--hidden_dim", type=int, default=128, help="rnn hidden dimension")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
parser.add_argument(
    "--batch_size", type=int, default=64, help="batch size of the model"
)
parser.add_argument(
    "--gradient_clip", type=float, default=1.0, help="gradient clipping"
)
parser.add_argument(
    "--l2_penalty", type=float, default=0.0, help="l2 regularization coefficient"
)
parser.add_argument(
    "--curriculum_threshold",
    type=float,
    default=0.96,
    help="threshold accuracy for introducing a harder curriculum",
)
parser.add_argument(
    "--max_epochs", type=int, default=300000, help="number of sequences to train"
)
parser.add_argument(
    "--reset_hidden", action="store_true", help="reset the hidden state of the RNN"
)
parser.add_argument(
    "--model_type",
    type=str,
    default="rnn",
    help="model to use for training (rnn|lstm|epHopRnn|epHopDelay)",
)

parser.add_argument(
    "--experiment_name",
    type=str,
    default="binary_ccopy_test",
    help="reset the hidden state of the RNN",
)
parser.add_argument(
    "--experiment_version", type=int, default=-1, help="version of the experiment"
)

parser.add_argument("--gpu", action="store_true", help="use gpu")
parser.add_argument(
    "--no_curriculum_training", action="store_true", help="dont use curriculum training"
)

args = parser.parse_args()
print(vars(args))  # output the input arguments

input_dim = args.input_dim
hidden_dim = args.hidden_dim
output_dim = input_dim
sequence_length = args.seq_length
learning_rate = args.learning_rate
batch_size = args.batch_size
task_id = tuple(args.task_id)
assert len(task_id) == 3, "task_id must be a tuple of 3 elements"
gradient_clip = args.gradient_clip
max_epochs = args.max_epochs
seed = args.seed
reset_hidden = args.reset_hidden
folder_name = "{}_{}".format(args.experiment_name, args.model_type)
experiment_version = None if args.experiment_version < 0 else args.experiment_version

if "EXPERIMENT_OUTPUT_DIR" not in os.environ:
    print("EXPERIMENT_OUTPUT_DIR environment variable not set")
    sys.exit()

if args.gpu:
    accelerator = "gpu"
else:
    accelerator = "auto"

torch.manual_seed(seed)
np.random.seed(seed)
seed_everything(seed)

if args.model_type == "rnn":
    model = RNNModel(input_dim, hidden_dim, output_dim, bias=False)
elif args.model_type == "lstm":
    model = LSTMModel(input_dim, hidden_dim, output_dim, bias=False)
else:
    print("Model type {} not recognized".format(args.model_type))
    sys.exit()

lightning_module = BinaryLinearVBTask(
    model,
    seed=seed,
    learning_rate=learning_rate,
    batch_size=batch_size,
    input_dim=input_dim,
    seq_length=sequence_length,
    model_type=args.model_type,
    hidden_dim=hidden_dim,
    task_id=task_id,
    curriculum=not args.no_curriculum_training,
    curriculum_threshold=args.curriculum_threshold,
    l2_penalty=args.l2_penalty,
)

print(
    "=============Logs saved in path: {}".format(
        os.path.join(os.environ["EXPERIMENT_OUTPUT_DIR"], folder_name)
    )
)

logger = TensorBoardLogger(
    save_dir=os.path.join(os.environ["EXPERIMENT_OUTPUT_DIR"], folder_name),
    log_graph=True,
    version=experiment_version,
    name="binary_linearVB",
)

checkpoint_callback = ModelCheckpoint(
    monitor="accuracy",
    save_top_k=5,
    mode="max",
    filename="exemplar-{epoch:02d}-{global_step}-{accuracy}",
)

trainer = Trainer(
    logger=logger,
    gradient_clip_val=gradient_clip,
    max_epochs=max_epochs,
    deterministic=True,
    accelerator=accelerator,
    num_sanity_val_steps=0,
    default_root_dir=os.path.join(os.environ["EXPERIMENT_OUTPUT_DIR"], folder_name),
)

trainer.fit(lightning_module)
