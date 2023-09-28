#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -t 8:00:00  # Job time limit
#SBATCH -o logs/6_1_linearVB/1_0_lvb-%A_%a.out
#SBATCH -e logs/6_1_linearVB/1_0_lvb-%A_%a.err
#SBATCH --array=1-20
#SBATCH -G 1

module add cuda/11.3.1
module add cudnn/cuda11-8.4.1.50

# load profile
source /home/.bashrc

conda init
conda activate /home/anaconda3/envs/func_mod_env

################## VERIFY THIS PRIOR TO RUNNING ##################

cd variable_binding_episodic/experiments/6_1_all_linear
EXPERIMENT_ID=0
EXPERIMENT_NAME="6_1_all_linear"
#################################################################

echo "Initializations Complete"

## v4
TASK_ID1=(255 9241421688590303745 9241421688590303745 9241421688590303745)
TASK_ID2=(0 0 262676 38637)
TASK_ID3=(0 0 189 145)

for SEED in $(seq 1 20);
do
  for TASK_ID in 0 1 2 3 4
  do
    for HIDDEN_DIM in 128 64 32
    do
      for L2_PENALTY in 0.0 0.001 0.1 10
      do
        for CURRICULUM_THRESHOLD in 0.9 0.98
        do
          echo "$SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_MAX $EXPERIMENT_ID"
          if [ $(( $EXPERIMENT_ID % $SLURM_ARRAY_TASK_MAX )) -eq $SLURM_ARRAY_TASK_ID  ]
          then
            echo "executing..."
            python 1_0_binary_linearVB.py --seed $SEED \
                                           --hidden_dim $HIDDEN_DIM \
                                           --input_dim 8 \
                                           --seq_length 8 \
                                           --experiment_name $EXPERIMENT_NAME \
                                           --experiment_version $EXPERIMENT_ID \
                                           --batch_size 64 \
                                           --max_epochs 10 \
                                           --model_type rnn \
                                           --task_id ${TASK_ID1[$TASK_ID]} ${TASK_ID2[$TASK_ID]} ${TASK_ID3[$TASK_ID]} \
                                           --l2_penalty $L2_PENALTY \
                                           --curriculum_threshold $CURRICULUM_THRESHOLD \
                                           --gpu
          fi
          ((EXPERIMENT_ID=EXPERIMENT_ID+1))
        done
      done
      done
    done
done
