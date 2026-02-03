#!/bin/bash
#SBATCH --job-name=no_mlp_12L_16H
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=A100-4h
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --requeue

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Previous SLURM_JOB_ID (if requeued): ${SLURM_RESTART_COUNT:-0}"

# Navigate to the nanoGPT directory
cd /home/nlp/matan_avitan/git/no_mlp_exp/nanoGPT

# Check if checkpoint exists to resume from
OUT_DIR="out_no_mlp_12L_16H"
if [ -f "${OUT_DIR}/ckpt.pt" ]; then
    echo "Found existing checkpoint, resuming training..."
    # Override init_from to resume
    python -c "
import sys
sys.path.insert(0, '.')
exec(open('config/train_no_mlp_12L_16H.py').read())
init_from = 'resume'
# Save modified config
with open('config/train_no_mlp_12L_16H_resume.py', 'w') as f:
    for k, v in globals().items():
        if not k.startswith('_') and k not in ['sys']:
            if isinstance(v, str):
                f.write(f'{k} = \"{v}\"\n')
            else:
                f.write(f'{k} = {v}\n')
"
    CONFIG_FILE="config/train_no_mlp_12L_16H_resume.py"
else
    echo "No checkpoint found, starting from scratch..."
    CONFIG_FILE="config/train_no_mlp_12L_16H.py"
fi

# Run training with DDP on 2 GPUs
echo "Using config: $CONFIG_FILE"
torchrun --standalone --nproc_per_node=2 train.py "$CONFIG_FILE"

echo "Job finished at $(date)"
