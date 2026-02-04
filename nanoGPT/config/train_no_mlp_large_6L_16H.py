# Large No-MLP transformer configuration
# d_mlp = 8192, n_embd = 2048, n_head = 16
# This is a larger variant of the No-MLP architecture

out_dir = "out_no_mlp_large_6L_16H"
eval_interval = 2000
eval_iters = 200
log_interval = 1
always_save_checkpoint = True

# wandb logging
wandb_log = True
wandb_project = "no_mlp_exp"
wandb_run_name = "no_mlp_large_6L_16H_d8192"

# Model configuration
n_layer = 6
n_head = 16
n_embd = 2048
block_size = 1024
bias = False
dropout = 0.0

# No-MLP specific settings
use_no_mlp = True
value_dim = 8192  # d_mlp = 8192

# Data
dataset = "openwebtext"
batch_size = 4  # Reduced for larger model
gradient_accumulation_steps = 120  # Effective batch size = 4 * 120 = 480

# Optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# System
device = "cuda"
dtype = "bfloat16"
compile = True

# Initialization - use resume for checkpointing in case of requeue
init_from = "scratch"  # Will be overridden by checkpoint if exists
