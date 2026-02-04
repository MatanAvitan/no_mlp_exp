# 26-layer, 16-head vanilla transformer configuration (standard MLP)
# n_embd = 1280, d_mlp = 4 * n_embd = 5120
# Tuned for local 4x A100

out_dir = "out_vanilla_26L_16H_1280"
eval_interval = 2000
eval_iters = 200
log_interval = 1
always_save_checkpoint = True

# wandb logging
wandb_log = True
wandb_project = "no_mlp_exp"
wandb_run_name = "vanilla_26L_16H_mlp4_w1280_4gpu"

# Model configuration
n_layer = 26
n_head = 16
n_embd = 1280
block_size = 1024
bias = False
dropout = 0.0

# Vanilla MLP settings
use_no_mlp = False
mlp_ratio = 4.0

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
