#!/bin/bash
#SBATCH --job-name=vanilla_gpt
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=p_b200_nlp
#SBATCH --account=ug_goldberg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Activate b200 conda environment
export PATH="/home/nlp/matan_avitan/.conda/envs/b200/bin:$PATH"

# Navigate to the nanoGPT directory
cd /home/nlp/matan_avitan/git/no_mlp_exp/nanoGPT

# Run training on single GPU
python train.py config/train_vanilla_6L_12H.py

echo "Job finished at $(date)"
