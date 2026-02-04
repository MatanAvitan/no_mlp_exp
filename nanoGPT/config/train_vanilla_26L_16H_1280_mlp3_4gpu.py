# 26-layer, 16-head vanilla transformer configuration (parameter-matched)
# n_embd = 1280, d_mlp = 3 * n_embd = 3840
# Tuned for local 4x A100

out_dir = "out_vanilla_26L_16H_1280_mlp3"
eval_interval = 2000
eval_iters = 200
log_interval = 1
always_save_checkpoint = True

# wandb logging
wandb_log = True
wandb_project = "no_mlp_exp"
wandb_run_name = "vanilla_26L_16H_mlp3_w1280_4gpu"

# Model configuration
n_layer = 26
n_head = 16
n_embd = 1280
block_size = 1024
bias = False
dropout = 0.0

# Vanilla MLP settings (parameter-matched)
use_no_mlp = False
mlp_ratio = 3.0

# Data - Optimized for 80GB A100s
dataset = "openwebtext"
batch_size = 24
gradient_accumulation_steps = (
    4  # Effective batch size = 24 * 4 * 4 = 384 (must be divisible by 4 GPUs)
)

# Optimizer
learning_rate = 2e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 2e-5  # learning_rate / 10

# System
device = "cuda"
dtype = "bfloat16"
compile = True

# Initialization - use resume for checkpointing in case of restart
init_from = "scratch"  # Will be overridden by checkpoint if exists
