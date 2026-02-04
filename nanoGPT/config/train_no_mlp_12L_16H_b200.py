# 12-layer, 16-head No-MLP transformer configuration for B200
# n_embd = 1024, value_dim = 4096
# Optimized for B200's 183GB GPU memory

out_dir = "out_no_mlp_12L_16H"
eval_interval = 2000
eval_iters = 200
log_interval = 1
always_save_checkpoint = True

# wandb logging
wandb_log = True
wandb_project = "no_mlp_exp"
wandb_run_name = "no_mlp_12L_16H_d4096"

# Model configuration
n_layer = 12
n_head = 16
n_embd = 1024
block_size = 1024
bias = False
dropout = 0.0

# No-MLP specific settings
use_no_mlp = True
value_dim = 4096  # 4 * n_embd = 4096

# Data - INCREASED for B200's 183GB memory
# With 177M params and bfloat16, we can fit much larger batches
dataset = "openwebtext"
batch_size = 32  # Increased from 8 for B200's large memory
gradient_accumulation_steps = (
    15  # Effective batch size = 32 * 15 = 480 (same as before)
)

# Optimizer
learning_rate = 3e-4  # Standard for ~180M parameter models
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 3e-5  # learning_rate / 10

# System
device = "cuda"
dtype = "bfloat16"
compile = True

# Initialization - use resume for checkpointing in case of requeue
init_from = "scratch"  # Will be overridden by checkpoint if exists
