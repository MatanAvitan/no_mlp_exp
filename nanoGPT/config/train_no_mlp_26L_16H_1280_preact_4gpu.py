# 26-layer, 16-head No-MLP transformer with pre-attention GELU (~0.5B params)
# n_embd = 1280, value_dim = 5120
# GELU applied to values BEFORE attention (mirrors MLP structure)
# Lower LR to prevent divergence seen in post-attention variant

out_dir = "out_no_mlp_26L_16H_1280_preact"
eval_interval = 2000
eval_iters = 200
log_interval = 1
always_save_checkpoint = True

# wandb logging
wandb_log = True
wandb_project = "no_mlp_exp"
wandb_run_name = "no_mlp_26L_16H_preact_d5120_w1280_4gpu"

# Model configuration
n_layer = 26
n_head = 16
n_embd = 1280
block_size = 1024
bias = False
dropout = 0.0

# No-MLP specific settings
use_no_mlp = True
value_dim = 5120  # 4 * n_embd
pre_attn_activation = True  # GELU before attention

# Data - Optimized for 80GB A100s
dataset = "openwebtext"
batch_size = 46
gradient_accumulation_steps = (
    4  # Effective batch size = 46 * 4 = 184 sequences (must be divisible by 4 GPUs)
)

# Optimizer - lower LR to prevent divergence
learning_rate = 6e-5
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-6  # learning_rate / 10

# System
device = "cuda"
dtype = "bfloat16"
compile = True

# Initialization - use resume for checkpointing in case of restart
init_from = "scratch"  # Will be overridden by checkpoint if exists
