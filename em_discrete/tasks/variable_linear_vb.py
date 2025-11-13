import os

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class BinaryLinearVarVBDataset(IterableDataset):
    """
    Binary and Linear Variable Binding Task with Variable Sequence Length support.

    Each task is identified by a tuple of 3 numbers assigned to it:
    (task_id_1, task_id_2, task_id_3)

    task_id_1 is the id for the power set enumeration of { 1, 2, 3, ... nd }.
    note that task_id_1 has to have d ones in its binary representation.

    task_id_2 is converted to a d-ary number encoding the combination of the choice obtained from the power set
    note that task_id_2 has to also satisfy certain conditions on the d-ary number. meaning the ith digit should be
    less than d-i.

    task_id_3 is binarized and represents the presence of absence of the sign in each element of the combination

    Note: id for repeat copy is always (2^{d}-1, 0, 0)

    Variable Length Support:
    - Set seq_length to a tuple (min, max) for variable-length sequences
    - When use_eos=True, an EOS (End-of-Sequence) marker is added as an extra input channel
    - The EOS marker is set to -1 at the timestep immediately after the input sequence ends
    - This helps the model learn when to transition from reading inputs to producing outputs
    - Using -1 (instead of +1 or 0) maintains consistency with the binary {-1, +1} input values

    Dimensions:
    - Data dimension: d (the actual task dimension)
    - Input dimension: d+1 if use_eos=True, else d
    - Output/target dimension: always d (EOS channel not included in targets)
    """

    def __init__(
        self,
        seed=0,
        d=5,
        seq_length=10,
        batch_size=64,
        horizon=36,
        task_id=(31, 0, 0),
        generator_stop=2000,
        use_eos=True,
    ):
        self.d = d
        self.batch_size = batch_size
        self.horizon = horizon
        self.use_eos = use_eos

        # Support variable sequence length: either int or tuple (min, max)
        if isinstance(seq_length, tuple):
            self.seq_length_range = seq_length
            self.seq_length = seq_length[0]  # Initialize with min length
            self.variable_length = True
        else:
            self.seq_length = seq_length
            self.seq_length_range = None
            self.variable_length = False

        self.data_d = self.d
        # Input dimension includes EOS channel if enabled
        self.input_d = self.d + 1 if self.use_eos else self.d
        self.composition_size = d // self.seq_length
        self.generator_stop = generator_stop
        self.task_id = task_id

        # Initialize with default seq_length for now
        self.f_operator = self.get_linear_operator(task_id, self.seq_length)

        self.set_horizon(horizon)

        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)

    def set_horizon(self, horizon):
        self.horizon = horizon
        self.final_seq_length = self.seq_length + self.horizon

    def get_linear_operator(self, task_id, seq_length):
        task_id1, task_id2, task_id3 = task_id

        ## Step 1: convert task_id1 to the choice of dimensions
        # convert the binary representation of task_id1 to the choice of dimensions
        choice1 = []
        ndimensions = 0
        temp_task_id1 = task_id1
        while temp_task_id1:
            if temp_task_id1 & 1:
                choice1.append(ndimensions)
            temp_task_id1 >>= 1
            ndimensions += 1

        assert len(choice1) == self.d, (
            "task_id1 must have d ones in its binary representation. "
            "Check hamming weight of task_id1. Expected: {} Found: {} "
            "for task_id: {}".format(self.d, len(choice1), task_id)
        )

        ## Step 2: convert task_id2 to the combination of the choice
        choice_d = 0
        temp_task_id2 = task_id2
        while temp_task_id2:
            digit = temp_task_id2 % self.d
            if digit > 0:
                assert digit < self.d - choice_d, (
                    "task_id2 must have the ith digit greater than d-i. "
                    "Check the choice of dimensions in task_id2"
                )
                # swap choice1[choice_d] and choice1[choice_d + digit]
                choice1[choice_d], choice1[choice_d + digit] = (
                    choice1[choice_d + digit],
                    choice1[choice_d],
                )

            temp_task_id2 //= self.d
            choice_d += 1

        ## Step 3: convert task_id3 to the sign of the choice
        signs = [1] * self.d
        choice_d = 0
        temp_task_id3 = task_id3
        while temp_task_id3:
            digit = temp_task_id3 % 2
            if digit == 1:
                signs[choice_d] = -1
            temp_task_id3 //= 2
            choice_d += 1

        ## Step 4: create the linear operator
        linear_operator = torch.zeros((seq_length * self.d, self.d))
        for d, choice_d in enumerate(choice1):
            linear_operator[choice_d, d] = signs[d]

        return linear_operator

    def convert_from_binary(self, x):
        x = x.clone()
        x[x < 0] = 0
        mask = 2 ** torch.arange(self.data_d - 1, -1, -1).to(x.device, x.dtype)
        return torch.sum(mask * x, -1)

    def __len__(self):
        """Return the total number of batches across all workers.

        Note: PyTorch will warn about __len__ with IterableDataset and num_workers > 1,
        but we properly handle worker splitting in __iter__ to avoid duplicate data.
        """
        return self.generator_stop

    def __iter__(self):
        # Handle multi-process data loading
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading, return the full iterator
            iter_start = 0
            iter_end = self.generator_stop
        else:
            # Multi-process data loading, split workload
            per_worker = int(
                np.ceil(self.generator_stop / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.generator_stop)

            # Set different random seed for each worker to avoid duplicate data
            seed = self.rng.randint(0, 2**32 - 1) + worker_id
            self.rng = np.random.RandomState(seed)

        for _ in range(iter_end - iter_start):
            # Sample sequence length for this batch if variable length is enabled
            if self.variable_length:
                current_seq_length = self.rng.randint(
                    self.seq_length_range[0], self.seq_length_range[1] + 1
                )
                # Regenerate the linear operator for this sequence length
                f_operator = self.get_linear_operator(self.task_id, current_seq_length)
            else:
                current_seq_length = self.seq_length
                f_operator = self.f_operator

            current_final_seq_length = current_seq_length + self.horizon

            # Store current sequence length as an attribute for external access
            self.current_seq_length = current_seq_length

            # Create tensors with input_d channels (includes EOS if enabled)
            x_sample = torch.zeros(
                (current_final_seq_length, self.batch_size, self.input_d)
            )
            y_sample = torch.zeros((current_final_seq_length, self.batch_size, self.d))

            # Fill in the random binary data for the first d channels
            x_sample[:current_seq_length, :, : self.d] = (
                torch.randint(0, 2, size=(current_seq_length, self.batch_size, self.d))
                * 2
                - 1
            )

            # Add EOS marker at the position right after input sequence ends
            if self.use_eos:
                x_sample[
                    current_seq_length, :, self.d
                ] = -1  # EOS marker (-1 to match binary values)

            # y_sample only contains the data channels (no EOS)
            y_sample[:, :, :] = x_sample[:, :, : self.d]

            for step in range(self.horizon):
                composed_seq_id = current_seq_length + step

                u_t = None
                u_t = y_sample[
                    composed_seq_id - current_seq_length : composed_seq_id, :, :
                ]
                u_t = torch.swapaxes(u_t, 0, 1)
                u_t = u_t.reshape((self.batch_size, -1))

                u_t = u_t @ f_operator
                y_sample[composed_seq_id, :, :] = u_t

            # y_sample = y_sample * 2 - 1
            y_sample[:current_seq_length, :, :] = 0
            # x_sample[:current_seq_length, :, :] = x_sample[:current_seq_length, :, :] * 2 - 1  # convert to in +1/-1 space

            # Return sequence length as part of the batch for variable length support
            yield x_sample, y_sample, current_seq_length


class BinaryLinearVarVBTask(pl.LightningModule):
    def __init__(
        self,
        model,
        seed=0,
        learning_rate=1e-5,
        batch_size=32,
        input_dim=6,
        seq_length=5,
        model_type="rnn",
        poly_power=1.0,
        curriculum=True,
        l2_penalty=0.0,
        task_id=(31, 0, 0),
        curriculum_threshold=0.96,
        curriculum_horizons=None,
        use_eos=True,
        **kwargs,
    ):
        super(BinaryLinearVarVBTask, self).__init__()
        self.save_hyperparameters(ignore=["model"])

        self.seed = seed
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.poly_power = poly_power
        self.l2_penalty = l2_penalty
        self.curriculum_threshold = curriculum_threshold
        self.curriculum_horizons = curriculum_horizons
        self.use_eos = use_eos

        # splitting datasets here so training and validation do not overlap
        full_train_dataset = BinaryLinearVarVBDataset(
            seed=self.seed,
            d=self.input_dim,
            seq_length=self.seq_length,
            task_id=task_id,
            batch_size=batch_size,
            use_eos=use_eos,
        )
        full_val_dataset = BinaryLinearVarVBDataset(
            seed=self.seed,
            seq_length=self.seq_length,
            d=self.input_dim,
            task_id=task_id,
            batch_size=batch_size,
            use_eos=use_eos,
        )
        full_test_dataset = BinaryLinearVarVBDataset(
            seed=self.seed,
            d=self.input_dim,
            seq_length=self.seq_length,
            task_id=task_id,
            batch_size=batch_size,
            use_eos=use_eos,
        )
        self.train_dataset = full_train_dataset
        self.val_dataset = full_val_dataset
        self.test_dataset = full_test_dataset

        # Validate model dimensions
        expected_input_dim = self.input_dim + 1 if use_eos else self.input_dim
        if hasattr(model, "input_dim") and model.input_dim != expected_input_dim:
            print(
                f"WARNING: Model input_dim ({model.input_dim}) does not match expected "
                f"dimension ({expected_input_dim}). With use_eos={use_eos}, model should "
                f"have input_dim={expected_input_dim} and output_dim={self.input_dim}"
            )

        # curiculum learning parameters
        self.curriculum = curriculum
        if self.curriculum_horizons is None:
            self.curriculum_horizons = [
                5,
                10,
                15,
                20,
                25,
                30,
                35,
                40,
                45,
                50,
                60,
                70,
                80,
                90,
                100,
            ]  # More gradual progression, especially around 40-60 range
        self.current_curriculum = 0
        self.best_curriculum_accuracy = (
            0.0  # Track best accuracy at current curriculum level
        )
        self.epochs_at_current_curriculum = 0  # Track epochs at current level

        if self.curriculum:
            self.train_dataset.set_horizon(
                self.curriculum_horizons[self.current_curriculum]
            )

        # uncomment this to make the computational graph in tensorboard
        # self.example_input_array = torch.rand((self.seq_length, self.batch_size, self.input_dim + 2))

    def forward(self, x):
        return self.model(x)

    def on_train_start(self) -> None:
        print("Logging computational graph")
        self.model.device = self.device
        self.model.initialize_hidden(self.batch_size, device=self.device)

    def training_step(self, batch, batch_nb):
        # Unpack batch - includes sequence length for variable length support
        if len(batch) == 3:
            x, y, seq_length = batch
        else:
            x, y = batch
            # Use self.seq_length if it's an integer, otherwise use the min value
            seq_length = (
                self.seq_length
                if isinstance(self.seq_length, int)
                else self.seq_length[0]
            )

        x = x.float().squeeze()
        y = y.float().squeeze()

        self.model.initialize_hidden(batch_size=self.batch_size, device=self.device)
        y_hat = self.forward(x)

        y_hat = y_hat.reshape((-1, self.input_dim))
        y_hat = y_hat.reshape((-1, self.batch_size, self.input_dim))
        y_hat = y_hat[seq_length:, :, :]
        y_hat = y_hat.reshape((-1, self.input_dim))
        y = y[seq_length:, :, :]
        y = y.reshape((-1, self.input_dim))

        loss = (y - y_hat) ** 2  # use the mse loss for
        loss = torch.mean(loss)

        # compute accuracy on only the y section with output
        y_hat = y_hat.reshape((-1, self.batch_size, self.input_dim))
        # y_hat = y_hat[self.train_dataset.STREAM.EVALUATION_FROM.value:, :, :]
        y_hat = y_hat.reshape((-1, self.input_dim))
        y = y.reshape((-1, self.batch_size, self.input_dim))
        # y = y[self.train_dataset.STREAM.EVALUATION_FROM.value:, :, :]
        y = y.reshape((-1, self.input_dim))

        # convert predictions
        y_hat_predictions = y_hat.detach().clone()
        y_hat_predictions[y_hat_predictions >= 0] = 1
        y_hat_predictions[y_hat_predictions < 0] = -1
        y_hat_predictions = y_hat_predictions.long()

        y = y.cpu().detach().numpy()
        y_hat_predictions = y_hat_predictions.cpu().detach().numpy()

        accuracy = (y_hat_predictions.astype(np.int_) == y.astype(np.int_)).astype(
            np.int_
        )
        accuracy = np.mean(accuracy)

        # Track best accuracy and epochs at current curriculum level
        if accuracy > self.best_curriculum_accuracy:
            self.best_curriculum_accuracy = accuracy

        self.epochs_at_current_curriculum += 1

        # Detect if model has collapsed (loss near 1.0, accuracy near 0.5)
        has_collapsed = loss.item() > 0.95 and accuracy < 0.55

        # Only advance curriculum if:
        # 1. Accuracy exceeds threshold
        # 2. Model hasn't collapsed
        # 3. Have trained for at least 5 epochs at this level (to ensure stability)
        if accuracy > self.curriculum_threshold and not has_collapsed:
            if (
                self.curriculum
                and self.current_curriculum < len(self.curriculum_horizons) - 1
                # and self.epochs_at_current_curriculum >= 5  # DISABLED: Test if this is causing slowdown
            ):
                self.current_curriculum += 1
                # truncate curriculum
                self.current_curriculum = min(
                    self.current_curriculum, len(self.curriculum_horizons) - 1
                )
                self.train_dataset.set_horizon(
                    self.curriculum_horizons[self.current_curriculum]
                )
                # Reset tracking variables for new curriculum level
                self.best_curriculum_accuracy = 0.0
                self.epochs_at_current_curriculum = 0
                print(
                    "Curriculum difficulty increased to: {}/{} (horizon={})".format(
                        self.current_curriculum + 1,
                        len(self.curriculum_horizons),
                        self.curriculum_horizons[self.current_curriculum],
                    )
                )

        # Warn if model appears to have collapsed
        if has_collapsed and self.epochs_at_current_curriculum > 10:
            print(
                f"WARNING: Model may have collapsed at curriculum {self.current_curriculum + 1}/{len(self.curriculum_horizons)} "
                f"(loss={loss.item():.3f}, acc={accuracy:.3f}). Consider restarting from checkpoint."
            )

        self.log("train_loss", loss.cpu().item(), prog_bar=True, sync_dist=True)
        self.log("train_acc", accuracy, prog_bar=True, sync_dist=True)

        logs = {"loss": loss, "accuracy": accuracy}
        return logs

    def training_end(self, outputs):
        outputs["avg_train_accuracy"] = 0
        return outputs

    def _common_step(self, batch, batch_nb, step):
        # Unpack batch - includes sequence length for variable length support
        if len(batch) == 3:
            x, y, seq_length = batch
        else:
            x, y = batch
            # Use self.seq_length if it's an integer, otherwise use the min value
            seq_length = (
                self.seq_length
                if isinstance(self.seq_length, int)
                else self.seq_length[0]
            )

        x = x.float().squeeze()
        y = y.float().squeeze()

        self.model.initialize_hidden(batch_size=self.batch_size, device=self.device)
        y_hat = self.forward(x)

        y_hat = y_hat.reshape((-1, self.input_dim))
        y_hat = y_hat.reshape((-1, self.batch_size, self.input_dim))
        y_hat = y_hat[seq_length:, :, :]
        y_hat = y_hat.reshape((-1, self.input_dim))
        y = y[seq_length:, :, :]
        y = y.reshape((-1, self.input_dim))

        # Compute loss before converting to numpy
        loss = (y - y_hat) ** 2
        loss = torch.mean(loss)

        # convert predictions
        y_hat_predictions = y_hat.detach().clone()
        y_hat_predictions[y_hat_predictions >= 0] = 1
        y_hat_predictions[y_hat_predictions < 0] = -1
        y_hat_predictions = y_hat_predictions.long()

        y = y.cpu().detach().numpy()
        y_hat_predictions = y_hat_predictions.cpu().detach().numpy()

        accuracy = (y_hat_predictions.astype(np.int_) == y.astype(np.int_)).astype(
            np.int_
        )
        accuracy = np.mean(accuracy)

        self.log(f"{step}_loss", loss.cpu().item(), prog_bar=True, sync_dist=True)
        self.log(f"{step}_acc", accuracy, prog_bar=True, sync_dist=True)

        return {f"{step}_accuracy": accuracy}

    def test_step(self, batch, batch_nb):
        return self._common_step(batch, batch_nb, "test")

    def validation_step(self, batch, batch_nb):
        step_result = self._common_step(batch, batch_nb, "val")
        return step_result

    def test_end(self, outputs):
        avg_accuracy = torch.stack([x["test_accuracy"] for x in outputs]).mean()

        logs = {"test_accuracy": avg_accuracy}

        return {"test_accuracy": avg_accuracy, "log": logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.l2_penalty
        )
        # Use CosineAnnealingWarmRestarts to periodically reset learning rate
        # This helps escape local minima by periodically increasing LR
        # T_0=10 means first restart after 10 epochs, then doubles each time

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=10, T_mult=2, eta_min=1e-5
        # )

        # Old schedulers that didn't work well:
        # ReduceLROnPlateau - just keeps reducing when stuck, makes it worse
        # CosineAnnealingLR - fixed schedule, can't escape if stuck mid-training
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "epoch",
        #         "frequency": 1,
        #     },
        # }

        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset)

    def val_dataloader(self):
        return DataLoader(self.val_dataset)

    def test_dataloader(self):
        return DataLoader(self.test_dataset)


if __name__ == "__main__":
    dataset = BinaryLinearVarVBDataset(
        seq_length=(5, 15), d=5, batch_size=64, horizon=10
    )

    iterator = iter(dataset)
    for i in range(10):
        x_sample, y_sample, seq_len = next(iterator)

        print(f"Sample {i + 1}: seq_length={seq_len}, total_length={x_sample.size(0)}")
        print(f"  Input shape:  {x_sample.shape}")
        print(f"  Output shape: {y_sample.shape}")

        # Count non-zero positions in input (first seq_len positions should be non-zero)
        nonzero_count = (x_sample[:, 0, 0] != 0).sum().item()
        print(f"  Non-zero input positions: {nonzero_count}")
        print()
