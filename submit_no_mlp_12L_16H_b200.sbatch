#!/bin/bash
#SBATCH --job-name=no_mlp_12L_16H_b200
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=p_b200_nlp
#SBATCH --account=ug_goldberg
#SBATCH --gres=gpu:2
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

# Navigate to the nanoGPT directory
cd /home/nlp/matan_avitan/git/no_mlp_exp/nanoGPT

# Configuration
OUT_DIR="out_no_mlp_12L_16H"
WANDB_ID_FILE="${OUT_DIR}/wandb_run_id.txt"

# Generate or load WandB run ID for continuity across requeues
if [ -f "${WANDB_ID_FILE}" ]; then
    export WANDB_RUN_ID=$(cat "${WANDB_ID_FILE}")
    echo "Loaded existing WandB run ID: ${WANDB_RUN_ID}"
else
    # Generate new run ID
    export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
    mkdir -p "${OUT_DIR}"
    echo "${WANDB_RUN_ID}" > "${WANDB_ID_FILE}"
    echo "Generated new WandB run ID: ${WANDB_RUN_ID}"
fi
export WANDB_RESUME=allow

# Check if checkpoint exists to resume from
if [ -f "${OUT_DIR}/ckpt.pt" ]; then
    echo "Found existing checkpoint, resuming training..."
    # Create resume config
    python -c "
import sys
sys.path.insert(0, '.')
exec(open('config/train_no_mlp_12L_16H_b200.py').read())
init_from = 'resume'
with open('config/train_no_mlp_12L_16H_b200_resume.py', 'w') as f:
    for k, v in globals().items():
        if not k.startswith('_') and k not in ['sys']:
            if isinstance(v, str):
                f.write(f'{k} = \"{v}\"\n')
            else:
                f.write(f'{k} = {v}\n')
"
    CONFIG_FILE="config/train_no_mlp_12L_16H_b200_resume.py"
else
    echo "No checkpoint found, starting from scratch..."
    CONFIG_FILE="config/train_no_mlp_12L_16H_b200.py"
fi

# Run training with DDP on 2 GPUs
echo "Using config: $CONFIG_FILE"
echo "WandB Run ID: ${WANDB_RUN_ID}"
torchrun --standalone --nproc_per_node=2 train.py "$CONFIG_FILE"

echo "Job finished at $(date)"
