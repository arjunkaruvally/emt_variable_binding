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
    BASE_EXPERIMENT_NAME="infinite_horizon_${NUM_MODELS}models_seqlen_${SEQ_LENGTH}_hidden_${HIDDEN_DIM}"

    echo "=========================================="
    echo "Starting experiment batch: $BASE_EXPERIMENT_NAME"
    echo "Hidden dimension: $HIDDEN_DIM"
    echo "Training $NUM_MODELS models with different seeds"
    echo "=========================================="

    # Loop through seeds (0, 1000, 2000, ..., 29000)
    for MODEL_IDX in $(seq 0 $((NUM_MODELS - 1))); do
        SEED=$((MODEL_IDX * 1000))
        MODEL_NUM=$((MODEL_IDX + 1))
        EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}/model-${MODEL_NUM}"

        echo ""
        echo "Training model $MODEL_NUM/$NUM_MODELS (seed=$SEED, hidden=$HIDDEN_DIM)"

        python train.py \
            --experiment_name "$EXPERIMENT_NAME" \
            --output_dir "$OUTPUT_DIR" \
            --seq_length $SEQ_LENGTH \
            --hidden_dim $HIDDEN_DIM \
            --seed $SEED \
            --epochs $EPOCHS \
            --learning_rate $LEARNING_RATE \
            --batch_size $BATCH_SIZE \
            --infinite_horizon \
            --output_horizon 15 \
            --total_length 300 \
            --task_id 255 0 0

        if [ $? -eq 0 ]; then
            echo "✓ Model $MODEL_NUM completed successfully"
        else
            echo "✗ Model $MODEL_NUM failed"
            exit 1
        fi
    done

    echo ""
    echo "=========================================="
    echo "✓ Completed all models for hidden_dim=$HIDDEN_DIM"
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "All experiments completed successfully!"
echo "=========================================="
echo ""
echo "Results saved in:"
for HIDDEN_DIM in 8 16 24 32 40 48 56 64 72; do
    echo "  - $OUTPUT_DIR/infinite_horizon_${NUM_MODELS}models_seqlen_${SEQ_LENGTH}_hidden_${HIDDEN_DIM}/"
done
echo ""
