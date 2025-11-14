import os
import sys

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class BinaryLinearVBInfiniteHorizonDataset(IterableDataset):
    """
    Binary and Linear Variable Binding Task with Infinite Horizon.

    This dataset generates a continuous stream of variable-length sequences separated by EOS markers.
    The model must learn to:
    1. Read sequences of varying lengths
    2. Detect EOS markers to know when a sequence ends
    3. Output the composed results for each sequence after seeing the EOS marker

    Structure:
    - Total sequence length: `total_length` timesteps
    - Multiple subsequences of variable length (sampled from `seq_length_range`)
    - Each subsequence followed by an EOS marker
    - After EOS, output `output_horizon` composed results
    - Then immediately start the next input subsequence

    Example timeline:
        t=0-4:   Input seq1 (length 5) + EOS at t=5
        t=6-9:   Output compositions for seq1
        t=10-15: Input seq2 (length 6) + EOS at t=16
        t=17-21: Output compositions for seq2
        ...

    Dimensions:
    - Data dimension: d (the actual task dimension)
    - Input dimension: d (EOS encoded as all -1s within data dimensions)
    - Output/target dimension: d (same as input)

    EOS Encoding:
    - EOS is encoded as all -1s across all d dimensions
    - Regular data never has all -1s (at least one dimension is +1)
    - This reserves the all--1s pattern exclusively for EOS
    """

    def __init__(
        self,
        seed=0,
        d=5,
        seq_length_range=(5, 10),
        batch_size=64,
        output_horizon=10,
        total_length=500,
        task_id=(31, 0, 0),
        generator_stop=500,  # Reduced from 2000: infinite horizon sequences are 37x longer
    ):
        self.d = d
        self.batch_size = batch_size
        self.seq_length_range = seq_length_range
        self.output_horizon = output_horizon
        self.total_length = total_length
        self.task_id = task_id
        self.generator_stop = generator_stop

        # Input dimension same as data dimension (EOS encoded within data)
        self.input_d = self.d
        self.data_d = self.d

        # Get the linear operator (use max seq_length for initialization)
        self.f_operator = self.get_linear_operator(task_id, seq_length_range[1])

        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)

    def get_linear_operator(self, task_id, seq_length):
        task_id1, task_id2, task_id3 = task_id

        ## Step 1: convert task_id1 to the choice of dimensions
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

    def __len__(self):
        return self.generator_stop

    def __iter__(self):
        # Handle multi-process data loading
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = self.generator_stop
        else:
            per_worker = int(
                np.ceil(self.generator_stop / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.generator_stop)

            seed = self.rng.randint(0, 2**32 - 1) + worker_id
            self.rng = np.random.RandomState(seed)

        for _ in range(iter_end - iter_start):
            # Create tensors for the entire sequence
            x_sample = torch.zeros((self.total_length, self.batch_size, self.input_d))
            y_sample = torch.zeros((self.total_length, self.batch_size, self.d))

            # Track positions and sequence information
            current_pos = 0
            sequences_info = []  # Store (start, length, eos_pos) for each sequence

            # Fill the timeline with multiple sequences
            while current_pos < self.total_length:
                # Sample sequence length
                seq_len = self.rng.randint(
                    self.seq_length_range[0], self.seq_length_range[1] + 1
                )

                # Check if we have room for: sequence + EOS + at least some outputs
                needed_space = seq_len + 1 + min(self.output_horizon, seq_len)
                if current_pos + needed_space > self.total_length:
                    break

                # Generate the linear operator for this sequence length
                f_operator = self.get_linear_operator(self.task_id, seq_len)

                # Fill in random binary input for this sequence
                seq_start = current_pos
                seq_end = current_pos + seq_len

                # Generate random binary data, ensuring it never creates all -1s pattern (reserved for EOS)
                # VECTORIZED: Generate all data for this sequence at once
                seq_data = torch.randint(0, 2, size=(seq_len, self.batch_size, self.d)) * 2 - 1

                # Fix any all -1s patterns (reserved for EOS)
                # Check which (time, batch) positions have all -1s
                all_minus_one = (seq_data == -1).all(dim=-1)  # Shape: (seq_len, batch_size)
                if all_minus_one.any():
                    # For each position that's all -1s, flip a random dimension to +1
                    positions = all_minus_one.nonzero(as_tuple=False)
                    for pos in positions:
                        t_idx, b_idx = pos[0].item(), pos[1].item()
                        flip_idx = self.rng.randint(0, self.d)
                        seq_data[t_idx, b_idx, flip_idx] = 1

                # Assign to x_sample
                x_sample[seq_start:seq_end, :, :] = seq_data

                # Copy input sequence to y_sample (needed for recurrent composition)
                y_sample[seq_start:seq_end, :, :] = x_sample[seq_start:seq_end, :, :]

                # Add EOS marker right after the sequence (all -1s across all dimensions)
                eos_pos = seq_end
                if eos_pos < self.total_length:
                    x_sample[eos_pos, :, :] = -1  # EOS encoded as all -1s

                # Store sequence info
                sequences_info.append((seq_start, seq_len, eos_pos))

                # Compute outputs for this sequence (after EOS marker)
                # IMPORTANT: The composition is RECURRENT - each output depends on previous outputs!
                output_start = eos_pos + 1
                num_outputs = min(self.output_horizon, self.total_length - output_start)

                for step in range(num_outputs):
                    output_pos = output_start + step

                    # Get the window for composition: last seq_len positions from y_sample
                    # This creates recurrence: each output depends on previous outputs
                    window_end = output_pos
                    window_start = window_end - seq_len

                    # Extract window from y_sample (not x_sample!)
                    # y_sample contains: original inputs + accumulated outputs
                    u_t = y_sample[window_start:window_end, :, :]
                    u_t = torch.swapaxes(u_t, 0, 1)
                    u_t = u_t.reshape((self.batch_size, -1))

                    # Apply composition function
                    u_t = u_t @ f_operator

                    y_sample[output_pos, :, :] = u_t

                # Move to next sequence position (after outputs)
                current_pos = output_start + num_outputs

            # Return the full sequence
            yield x_sample, y_sample, sequences_info


class BinaryLinearVBInfiniteHorizonTask(pl.LightningModule):
    def __init__(
        self,
        model,
        seed=0,
        learning_rate=1e-5,
        batch_size=32,
        input_dim=6,
        seq_length_range=(5, 10),
        output_horizon=10,
        total_length=500,
        model_type="rnn",
        l2_penalty=0.0,
        task_id=(31, 0, 0),
        **kwargs,
    ):
        super(BinaryLinearVBInfiniteHorizonTask, self).__init__()
        self.save_hyperparameters(ignore=["model"])

        self.seed = seed
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.seq_length_range = seq_length_range
        self.output_horizon = output_horizon
        self.total_length = total_length
        self.l2_penalty = l2_penalty

        # Create datasets
        full_train_dataset = BinaryLinearVBInfiniteHorizonDataset(
            seed=self.seed,
            d=self.input_dim,
            seq_length_range=self.seq_length_range,
            batch_size=batch_size,
            output_horizon=output_horizon,
            total_length=total_length,
            task_id=task_id,
        )
        full_val_dataset = BinaryLinearVBInfiniteHorizonDataset(
            seed=self.seed + 1000,
            d=self.input_dim,
            seq_length_range=self.seq_length_range,
            batch_size=batch_size,
            output_horizon=output_horizon,
            total_length=total_length,
            task_id=task_id,
        )
        full_test_dataset = BinaryLinearVBInfiniteHorizonDataset(
            seed=self.seed + 2000,
            d=self.input_dim,
            seq_length_range=self.seq_length_range,
            batch_size=batch_size,
            output_horizon=output_horizon,
            total_length=total_length,
            task_id=task_id,
        )

        self.train_dataset = full_train_dataset
        self.val_dataset = full_val_dataset
        self.test_dataset = full_test_dataset

        # Validate model dimensions
        expected_input_dim = self.input_dim  # EOS encoded within data dimensions
        if hasattr(model, "input_dim") and model.input_dim != expected_input_dim:
            print(
                f"WARNING: Model input_dim ({model.input_dim}) does not match expected "
                f"dimension ({expected_input_dim}). Model should have "
                f"input_dim={expected_input_dim} and output_dim={self.input_dim}"
            )

    def forward(self, x):
        return self.model(x)

    def on_train_start(self) -> None:
        print("Logging computational graph")
        self.model.device = self.device
        self.model.initialize_hidden(self.batch_size, device=self.device)

    def training_step(self, batch, batch_nb):
        x, y, sequences_info = batch

        x = x.float().squeeze()
        y = y.float().squeeze()

        self.model.initialize_hidden(batch_size=self.batch_size, device=self.device)
        y_hat = self.forward(x)

        # Reshape predictions
        y_hat = y_hat.reshape((-1, self.input_dim))
        y_hat = y_hat.reshape((-1, self.batch_size, self.input_dim))

        # Compute loss only on positions where target is non-zero
        # (i.e., where we expect outputs)
        mask = (y.abs().sum(dim=-1) > 0).float()  # Shape: (time, batch)

        # Reshape targets
        y = y.reshape((-1, self.batch_size, self.input_dim))

        # Compute MSE loss with mask
        loss = ((y - y_hat) ** 2).mean(dim=-1)  # Shape: (time, batch)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)  # Average over valid positions

        # Compute accuracy on valid positions
        y_hat_predictions = y_hat.detach().clone()
        y_hat_predictions[y_hat_predictions >= 0] = 1
        y_hat_predictions[y_hat_predictions < 0] = -1
        y_hat_predictions = y_hat_predictions.long()

        y_np = y.cpu().detach().numpy()
        y_hat_np = y_hat_predictions.cpu().detach().numpy()
        mask_np = mask.cpu().detach().numpy()

        # Only compute accuracy where mask is active
        valid_positions = mask_np > 0
        if valid_positions.sum() > 0:
            accuracy = (
                (
                    y_hat_np[valid_positions].astype(np.int_)
                    == y_np[valid_positions].astype(np.int_)
                )
                .astype(np.int_)
                .mean()
            )
        else:
            accuracy = 0.0

        self.log("train_loss", loss.cpu().item(), prog_bar=True, sync_dist=True)
        self.log("train_acc", accuracy, prog_bar=True, sync_dist=True)

        return {"loss": loss, "accuracy": accuracy}

    def validation_step(self, batch, batch_nb):
        x, y, sequences_info = batch

        x = x.float().squeeze()
        y = y.float().squeeze()

        self.model.initialize_hidden(batch_size=self.batch_size, device=self.device)
        y_hat = self.forward(x)

        # Reshape predictions
        y_hat = y_hat.reshape((-1, self.input_dim))
        y_hat = y_hat.reshape((-1, self.batch_size, self.input_dim))

        # Compute loss only on positions where target is non-zero
        mask = (y.abs().sum(dim=-1) > 0).float()

        # Reshape targets
        y = y.reshape((-1, self.batch_size, self.input_dim))

        # Compute MSE loss with mask
        loss = ((y - y_hat) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        # Compute accuracy on valid positions
        y_hat_predictions = y_hat.detach().clone()
        y_hat_predictions[y_hat_predictions >= 0] = 1
        y_hat_predictions[y_hat_predictions < 0] = -1
        y_hat_predictions = y_hat_predictions.long()

        y_np = y.cpu().detach().numpy()
        y_hat_np = y_hat_predictions.cpu().detach().numpy()
        mask_np = mask.cpu().detach().numpy()

        # Only compute accuracy where mask is active
        valid_positions = mask_np > 0
        if valid_positions.sum() > 0:
            accuracy = (
                (
                    y_hat_np[valid_positions].astype(np.int_)
                    == y_np[valid_positions].astype(np.int_)
                )
                .astype(np.int_)
                .mean()
            )
        else:
            accuracy = 0.0

        self.log("val_loss", loss.cpu().item(), prog_bar=True, sync_dist=True)
        self.log("val_acc", accuracy, prog_bar=True, sync_dist=True)

        return {"val_loss": loss, "val_accuracy": accuracy}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.l2_penalty
        )
        # Use ReduceLROnPlateau for stable training (no periodic LR resets)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",  # Monitor training loss
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset)

    def val_dataloader(self):
        return DataLoader(self.val_dataset)

    def test_dataloader(self):
        return DataLoader(self.test_dataset)


if __name__ == "__main__":
    # Test the infinite horizon dataset
    dataset = BinaryLinearVBInfiniteHorizonDataset(
        seq_length_range=(5, 10),
        d=5,
        batch_size=4,
        output_horizon=10,
        total_length=100,
        generator_stop=3,
    )

    iterator = iter(dataset)
    for i in range(3):
        x_sample, y_sample, sequences_info = next(iterator)

        print(f"\nSample {i + 1}:")
        print(f"  Total length: {x_sample.shape[0]}")
        print(f"  Input shape:  {x_sample.shape}")
        print(f"  Output shape: {y_sample.shape}")
        print(f"  Number of sequences: {len(sequences_info)}")

        for j, (start, length, eos_pos) in enumerate(sequences_info):
            print(f"    Seq {j + 1}: start={start}, length={length}, eos={eos_pos}")

        # Check EOS markers (all -1s pattern)
        batch_idx = 0
        # EOS is where all dimensions are -1
        is_eos = torch.all(x_sample[:, batch_idx, :] == -1, dim=-1)
        eos_positions = is_eos.nonzero(as_tuple=True)[0]
        print(f"  EOS marker positions: {eos_positions.tolist()}")

        # Check where outputs are non-zero
        output_nonzero = (y_sample[:, batch_idx, :].abs().sum(dim=-1) > 0).nonzero(
            as_tuple=True
        )[0]
        print(
            f"  Non-zero output positions: {output_nonzero.tolist()[:20]}..."
        )  # Show first 20
