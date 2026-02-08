#!/bin/bash
#SBATCH --job-name=no_mlp_preact_lr2e4
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=p_b200_nlp
#SBATCH --account=ug_goldberg
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --requeue

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Previous SLURM_JOB_ID (if requeued): ${SLURM_RESTART_COUNT:-0}"

# Activate b200 conda environment
export PATH="/home/nlp/matan_avitan/.conda/envs/b200/bin:$PATH"

# Force local rendezvous to avoid hostname routing issues
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + SLURM_JOBID % 1000))
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# Ensure logs directory exists
mkdir -p /home/nlp/matan_avitan/git/no_mlp_exp/logs

# Navigate to the nanoGPT directory
cd /home/nlp/matan_avitan/git/no_mlp_exp/nanoGPT

OUT_DIR="out_no_mlp_26L_16H_1280_preact_lr2e4"
CONFIG_FILE="config/train_no_mlp_26L_16H_1280_preact_lr2e4_4gpu.py"

if [ -f "${OUT_DIR}/ckpt.pt" ]; then
    echo "Found existing checkpoint, resuming training..."
    torchrun --nnodes=1 --nproc_per_node=4 \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        train.py "$CONFIG_FILE" --init_from=resume
else
    echo "No checkpoint found, starting from scratch..."
    torchrun --nnodes=1 --nproc_per_node=4 \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        train.py "$CONFIG_FILE"
fi

echo "Job finished at $(date)"
