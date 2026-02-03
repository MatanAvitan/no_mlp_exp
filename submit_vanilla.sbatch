#!/bin/bash
#SBATCH --job-name=vanilla_gpt
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Navigate to the nanoGPT directory
cd /home/nlp/matan_avitan/git/no_mlp_exp/nanoGPT

# Run training with DDP on 2 GPUs
torchrun --standalone --nproc_per_node=2 train.py config/train_vanilla_6L_12H.py

echo "Job finished at $(date)"
