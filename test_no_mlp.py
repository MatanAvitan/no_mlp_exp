#!/usr/bin/env python3
"""Quick test of the No-MLP architecture"""

import sys

sys.path.insert(0, "/home/nlp/matan_avitan/git/no_mlp_exp/nanoGPT")

import torch
from model import GPTConfig, GPT

# Test with no-MLP architecture
print("Testing No-MLP architecture...")
config = GPTConfig(
    block_size=128,
    vocab_size=100,
    n_layer=2,
    n_head=4,
    n_embd=64,
    dropout=0.0,
    bias=False,
    use_no_mlp=True,
    value_dim=None,  # Will default to 4 * n_embd = 256
)

model = GPT(config)
print(f"Model created successfully!")
print(f"Number of parameters: {model.get_num_params() / 1e6:.2f}M")

# Test forward pass
batch_size = 2
seq_len = 10
x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
logits, loss = model(x)
print(f"Forward pass successful!")
print(f"Logits shape: {logits.shape}")
print(f"Loss: {loss}")

# Test with standard architecture for comparison
print("\nTesting standard architecture for comparison...")
config_std = GPTConfig(
    block_size=128,
    vocab_size=100,
    n_layer=2,
    n_head=4,
    n_embd=64,
    dropout=0.0,
    bias=False,
    use_no_mlp=False,
)

model_std = GPT(config_std)
print(f"Standard model created successfully!")
print(f"Number of parameters: {model_std.get_num_params() / 1e6:.2f}M")

logits_std, loss_std = model_std(x)
print(f"Standard forward pass successful!")
print(f"Logits shape: {logits_std.shape}")

print("\n✓ All tests passed!")
