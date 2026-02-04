# AGENTS.md - Guidelines for AI Coding Agents

This document provides guidelines for AI agents working on the No-MLP Transformer experiment codebase.

## Project Overview

Research project exploring a novel transformer architecture that removes the dedicated MLP layer and instead integrates MLP-like computation into the attention mechanism by:
- Expanding the value projection to `value_dim = 4 * n_embd`
- Applying GELU activation after attention weighting: `GELU(att @ v) @ W_o`
- Removing the separate MLP block entirely

## Experiment Goals

- Compare No-MLP attention against vanilla MLP baselines at ~0.5B parameters.
- Keep comparisons fair: same data, tokens seen, optimizer, LR schedule, sequence length, and eval protocol.
- Report params, MFU, tokens/sec, memory, and validation loss for each run.

## Key Files

| File | Purpose |
|------|---------|
| `nanoGPT/model.py` | Model with No-MLP architecture support |
| `nanoGPT/train.py` | Training script |
| `nanoGPT/config/train_no_mlp_6L_12H.py` | 6-layer, 12-head No-MLP config |
| `submit_no_mlp.sbatch` | SLURM submission script |

## Architecture Configuration

```python
# Enable No-MLP mode
use_no_mlp = True
value_dim = None  # Defaults to 4 * n_embd (3072 for n_embd=768)

# Vanilla MLP width control
mlp_ratio = 4.0  # hidden dim = mlp_ratio * n_embd

# Standard settings
n_layer = 6
n_head = 12
n_embd = 768
```

## Comparison Protocol

- Primary comparison: parameter-matched at fixed depth/width. For vanilla, use `mlp_ratio = 3.0` (so `d_mlp = 3 * n_embd`) when No-MLP uses `value_dim = 4 * n_embd`.
- Secondary comparison: standard vanilla with `mlp_ratio = 4.0` at the same depth/width.
- Optional compute-matched comparison: keep `mlp_ratio = 4.0` and equalize total FLOPs by adjusting steps or model size.
- Keep batch, tokens seen, `block_size`, optimizer, and eval settings identical across runs.
- Prefer head_dim-friendly widths (e.g., `n_embd=1280` with `n_head=16` gives head_dim=80).

## KV Cache Implications

- Expanded V acts as the first half of the MLP, so caching V reuses that compute across decoding steps.
- This can improve per-token latency at long contexts by removing the separate MLP and avoiding repeated V-side expansion.
- Tradeoff: larger KV cache increases memory footprint and bandwidth pressure; speedups depend on kernel efficiency.
- Best gains show up at long sequence lengths; short contexts can become bandwidth-bound.

## Recent Findings (Local A100 4x)

- 26L/16H/1280 No-MLP is ~490.6M params and reaches ~32% MFU with high micro-batch sizes.
- batch_size=46, grad_accum=4 sustains ~79GB/GPU on 80GB A100s without OOM.
- batch_size=48 OOMs during compilation (~+4.6GB needed).

## Compute Resources

### Resource Priority (Use in This Order)

**IMPORTANT**: Always use Slurm to submit jobs. Never SSH directly to execute code or access compute nodes (including B200).

1. **Slurm dgx-b200-01** - `p_b200_nlp` partition - 8x NVIDIA B200 (183GB each) - Fastest, use first, for this queue I can submit jobs with up to 4 GPUs.
2. **Slurm dgx-b200-01** - `p_b200_goldberg` partition - 8x NVIDIA B200 (183GB each) - Secondary priority, use if `p_b200_nlp` is full, here I can submit jobs with up to 1 GPU only.
3. **dsinlp01** (current server) - 8x NVIDIA A100-SXM4-80GB - Local, no queue - it doesnt work currently.
4. **Slurm H200** - `H200-4h` or `H200-12h` partitions on hpc8h200-01, use the 12-h queue for longer than 4 hours jobs, keep in mind that the b200 has much more GPU memory per core (183GB vs 80GB), so you might need to reduce batch size accordingly.
5. **Slurm A100** - `A100-4h` partition on hpc2a100-01
6. **dgx02-03** - Legacy DGX servers, use as fallback, you can SSH directly here and run the code on whatever GPU is free.

The idea is that you never should wait for a job to start, if the highest priority resource is busy, just go to the next one.

### Server Details

| Server | GPUs | Memory/GPU | Access | Notes |
|--------|------|------------|--------|-------|
| `dgx-b200-01` | 8x B200 | 183GB | Slurm only | Newest, fastest |
| `dsinlp01` | 8x A100-SXM4 | 80GB | Local (current) | Good for parallel runs |
| `hpc8h200-01` | 2x H200 | - | Slurm `H200-*` | Via Slurm only |
| `hpc2a100-01` | 2x A100 | - | Slurm `A100-4h` | Via Slurm only |
| `dgx02-03` | Varies | - | SSH | Also a great option for longer than 12 hours jobs |

## Slurm Cluster Usage

### SSH Access

Connect to the Slurm login node using the `dsinlp01_id_rsa` SSH key:

```bash
ssh -i ~/.ssh/dsinlp01_id_rsa slurm-login.lnx.biu.ac.il
```

### Available Partitions

| Partition | Nodes | Max Time | Max GPUs | Access (Accounts) | Node Memory | Notes |
|-----------|-------|----------|----------|-------------------|-------------|-------|
| **GPU Partitions** ||||
| `p_b200_nlp` | dgx-b200-01 | 4h | 4 | Prof. Tsarfaty, Dagan, Goldberg groups | 2TB | **Fastest**, 183GB per GPU per user |
| `p_b200_goldberg` | dgx-b200-01 | 4h | 1 | Prof. Goldberg's group only | 2TB | B200 secondary priority |
| `B200-4h` | dgx-b200-01, dgx-b200-02 | 4h | 2 | All users | 2TB | B200 for all users |
| `H200-4h` | hpc8h200-01 | 4h | 2 | All users | 2TB | H200 GPUs |
| `H200-12h` | hpc8h200-01 | 12h | 2 | All users | 2TB | H200 for longer jobs |
| `A100-4h` | hpc2a100-01, dsiasaf01 | 4h | 2 | All users | 512GB | A100 GPUs |

**Important B200/H200 Notes:**
- B200 max time is **4 hours** on all partitions
- Use `--requeue` flag for week-long training (job auto-resubmits after 4h)
- GPU quota per account: `p_b200_nlp` allows up to 4 GPUs per user
- Job may show `QOSGrpGRES` (pending) if GPU quota reached - it will start automatically when available
| `RTX6000-4h` | mm-lab02 | 4h | 2 | All users | 2TB | RTX 6000 GPUs |
| `L4-4h` | hpc8l4-01-01 | 4h | 2 | All users | 512GB | L4 GPUs |
| `L4-12h` | hpc8l4-01-01 | 12h | 2 | All users | 512GB | L4 for longer jobs |
| `generic` | dsicsgpu[02-09], dsisarit[02,05] | 4h | 16 total, 2 per job | All users | 192GB | General GPU jobs |
| `generic-48G` | dsiaw15 | 4h | 2 | All users | 128GB | Lower memory GPUs |
| **CPU Partitions** ||||
| `cpu1T-24h` | hpccpu01 | 24h | - | All users | 1TB | CPU jobs |
| `cpu192G-48h` | dml[02-25] | 48h | - | All users | 192GB | CPU jobs |
| `cpu512G-48h` | skittles, skittles[01-22] | 48h | - | All users | 512GB | CPU jobs |


### Submitting Jobs

```bash
# Submit job via SSH
ssh -i ~/.ssh/dsinlp01_id_rsa slurm-login.lnx.biu.ac.il "cd /home/nlp/matan_avitan/git/no_mlp_exp && sbatch submit_no_mlp.sbatch"
```

### Job Script Template

```bash
#!/bin/bash
#SBATCH --job-name=no_mlp_gpt
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

cd /home/nlp/matan_avitan/git/no_mlp_exp/nanoGPT
torchrun --standalone --nproc_per_node=2 train.py config/train_no_mlp_6L_12H.py
```

### Monitoring Jobs

```bash
# Use the monitor script (from this project directory)
cd /home/nlp/matan_avitan/git/no_mlp_exp && bash ~/slurm_monitor.sh

# Or specify logs directory explicitly
bash ~/slurm_monitor.sh -l /home/nlp/matan_avitan/git/no_mlp_exp/logs

# Check job queue
ssh -i ~/.ssh/dsinlp01_id_rsa slurm-login.lnx.biu.ac.il "squeue -u \$USER"

# View job details
ssh -i ~/.ssh/dsinlp01_id_rsa slurm-login.lnx.biu.ac.il "scontrol show job <job_id>"

# Cancel a job
ssh -i ~/.ssh/dsinlp01_id_rsa slurm-login.lnx.biu.ac.il "scancel <job_id>"

# Check logs directly
tail -50 /home/nlp/matan_avitan/git/no_mlp_exp/logs/slurm_*.out
```

### Important Notes

1. **Default resources**: Jobs get 1 CPU and 16GB RAM by default; specify more with `--cpus-per-task` and `--mem`
2. **GPU allocation**: Must explicitly request GPUs with `--gres=gpu:N`
3. **Time limits**: Jobs exceeding time limits are suspended and requeued
4. **Max jobs**: `generic` partition allows max 4 concurrent jobs per user
5. **Log files**: Output goes to `logs/slurm_<jobid>.out` and `.err`

## Running on dsinlp01 (Current Server)

```bash
# Run on specific GPU (0-7 available)
CUDA_VISIBLE_DEVICES=0 nohup python train.py config/train_no_mlp_6L_12H.py > ../logs/train.out 2>&1 &

# Check GPU usage
nvidia-smi
```

## Training Commands

### Prepare OpenWebText Data
```bash
cd nanoGPT/data/openwebtext && python prepare.py
```

### Train No-MLP Model (Single GPU)
```bash
cd nanoGPT
CUDA_VISIBLE_DEVICES=0 python train.py config/train_no_mlp_6L_12H.py
```

### Train No-MLP Model (Multi-GPU DDP)
```bash
cd nanoGPT
torchrun --standalone --nproc_per_node=2 train.py config/train_no_mlp_6L_12H.py
```

## Code Style

- **Imports**: stdlib, third-party (torch, numpy), then local
- **Naming**: PascalCase for classes, snake_case for functions/variables
- **Type hints**: Use for function signatures
- **Device handling**: `device = 'cuda' if torch.cuda.is_available() else 'cpu'`
- **BFloat16**: Use `dtype="bfloat16"` for A100/H200/B200 GPUs
