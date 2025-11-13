#!/bin/bash

# Batch training script for infinite horizon experiments
# - 30 models (different seeds) per hidden dimension
# - 20 epochs
# - 9 hidden dimensions: 8, 16, 24, 32, 40, 48, 56, 64, 72
# - Infinite horizon task with fixed sequence length 8

# Set output directory
OUTPUT_DIR="./results"

# Common parameters
NUM_MODELS=30
EPOCHS=20
SEQ_LENGTH=8
LEARNING_RATE=0.001
BATCH_SIZE=64
NUM_WORKERS=2

echo "=========================================="
echo "Starting batch training experiments"
echo "Infinite Horizon Mode - Fixed SeqLen 8"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Number of models per config: $NUM_MODELS"
echo "  - Epochs: $EPOCHS"
echo "  - Sequence length: $SEQ_LENGTH"
echo "  - Hidden dimensions: 8, 16, 24, 32, 40, 48, 56, 64, 72"
echo "  - Output directory: $OUTPUT_DIR"
echo ""

# Loop through hidden dimensions from 8 to 72 in steps of 8
for HIDDEN_DIM in 8 16 24 32 40 48 56 64 72; do
    EXPERIMENT_NAME="infinite_horizon_${NUM_MODELS}models_seqlen_${SEQ_LENGTH}_hidden_${HIDDEN_DIM}"

    echo "=========================================="
    echo "Starting experiment: $EXPERIMENT_NAME"
    echo "Hidden dimension: $HIDDEN_DIM"
    echo "=========================================="

    python train.py \
        --experiment_name "$EXPERIMENT_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --seq_length $SEQ_LENGTH \
        --hidden_dim $HIDDEN_DIM \
        --num_models $NUM_MODELS \
        --epochs $EPOCHS \
        --learning_rate $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --infinite_horizon \
        --output_horizon 15 \
        --total_length 300 \
        --task_id 255 0 0

    if [ $? -eq 0 ]; then
        echo ""
        echo "Successfully completed: $EXPERIMENT_NAME"
        echo ""
    else
        echo ""
        echo "Failed: $EXPERIMENT_NAME"
        echo ""
        exit 1
    fi
done

echo "All experiments completed successfully!"
echo ""
echo "Results saved in:"
for HIDDEN_DIM in 8 16 24 32 40 48 56 64 72; do
    echo "  - $OUTPUT_DIR/infinite_horizon_${NUM_MODELS}models_seqlen_${SEQ_LENGTH}_hidden_${HIDDEN_DIM}"
done
echo ""
