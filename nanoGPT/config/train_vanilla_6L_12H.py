# 6-layer, 12-head Vanilla transformer configuration (with MLP)
# This is the baseline for comparison with No-MLP architecture

out_dir = "out_vanilla_6L_12H"
eval_interval = 2000
eval_iters = 200
log_interval = 1
always_save_checkpoint = True

# wandb logging
wandb_log = True
wandb_project = "no_mlp_exp"
wandb_run_name = "vanilla_6L_12H_baseline"

# Model configuration
n_layer = 6
n_head = 12
n_embd = 768
block_size = 1024
bias = False
dropout = 0.0

# Vanilla settings (with MLP)
use_no_mlp = False
value_dim = None  # Not used when use_no_mlp=False

# Data
dataset = "openwebtext"
batch_size = 12
gradient_accumulation_steps = 40  # 5 * 8

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

# Initialization
init_from = "scratch"
