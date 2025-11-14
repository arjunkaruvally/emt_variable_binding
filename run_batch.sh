#!/bin/bash

#SBATCH -c 8  # Number of Cores per Task
#SBATCH --mem=24g  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -t 1-23:00:00  # Job time limit
#SBATCH -o /scratch4/workspace/akaruvally_umass_edu-simple/logs/varbind_1_lra-%A_%a.out
#SBATCH -e /scratch4/workspace/akaruvally_umass_edu-simple/logs/varbind_1_lra-%A_%a.err
#SBATCH --array=0-39
#SBATCH -G 2

#SBATCH --account=pi_erietman_umass_edu

# Batch training script for infinite horizon experiments
# - 30 models (different seeds) per hidden dimension
# - 20 epochs
# - 9 hidden dimensions: 8, 16, 24, 32, 40, 48, 56, 64, 72
# - Infinite horizon task with fixed sequence length 8

# Set output directory
OUTPUT_DIR="/scratch4/workspace/akaruvally_umass_edu-simple/output/varbinding"

hostname

module load cuda/12.1

# load profile
source /home/akaruvally_umass_edu/.bashrc
source /modules/apps/miniconda/4.8.3/etc/profile.d/conda.sh

conda activate /work/pi_erietman_umass_edu/arjun/envs/waveRNN

# Common parameters
NUM_MODELS=30
EPOCHS=20
SEQ_LENGTH=8
LEARNING_RATE=0.001
BATCH_SIZE=64

EXPERIMENT_NAME="infinite_horizon_${NUM_MODELS}models_seqlen_${SEQ_LENGTH}"

echo "=========================================="
echo "Starting batch training experiments"
echo "Infinite Horizon Mode - Fixed SeqLen"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Number of models per config: $NUM_MODELS"
echo "  - Epochs: $EPOCHS"
echo "  - Sequence length: $SEQ_LENGTH"
echo "  - Hidden dimensions: 8, 16, 24, 32, 40, 48, 56, 64, 72"
echo "  - Output directory: $OUTPUT_DIR"
echo ""

START_JOB_ID=0

################### Test the slurm script before submission
if [ -z ${SLURM_ARRAY_TASK_ID+x} ];
then
  echo "BASH: not a slurm array job"
  IN_SLURM_ARRAY=0;
  # simulated SLURM environment variables
  SLURM_ARRAY_TASK_ID=0;
  SLURM_ARRAY_TASK_COUNT=10;
  OUTPUT_DIR="${OUTPUT_DIR}_debug"
else
  echo "SLURM ARRAY JOB DETECTED";
  IN_SLURM_ARRAY=1;
fi
##################

JOB_ID=0  # This is the start of the jobid (identifies each job). CAUTION: this just numbers the job differently
GROUP_ID=0   # this is the start of the group id (group jobs by seed for averages)

# Loop through hidden dimensions from 8 to 72 in steps of 8
for HIDDEN_DIM in 8 16 24 32 40 48 56 64 72; do
    # Loop through seeds (0, 1000, 2000, ..., 29000)
    for MODEL_IDX in $(seq 0 $((NUM_MODELS - 1))); do
        SEED=$((MODEL_IDX * 1000))
        MODEL_NUM=$((MODEL_IDX + 1))

        echo "$SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT $JOB_ID / $LAST_JOB_ID"
        if [ $(( $JOB_ID % $SLURM_ARRAY_TASK_COUNT )) -eq $SLURM_ARRAY_TASK_ID  ]
        then
            EXPERIMENT_NAME="${EXPERIMENT_NAME}_hidden_${HIDDEN_DIM}"

            echo "=========================================="
            echo "Starting experiment batch: $EXPERIMENT_NAME"
            echo "Hidden dimension: $HIDDEN_DIM"
            echo "Training $NUM_MODELS models with different seeds"
            echo "=========================================="
        
            EXPERIMENT_NAME="${EXPERIMENT_NAME}/model-${MODEL_NUM}"

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
                echo ""
                echo "Successfully completed: $EXPERIMENT_NAME"
                echo ""
            else
                echo ""
                echo "Failed: $EXPERIMENT_NAME"
                echo ""
                exit 1
            fi
        fi
        ((JOB_ID=JOB_ID+1))
    done
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
