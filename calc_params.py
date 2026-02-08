import sys

sys.path.insert(0, "/home/nlp/matan_avitan/git/no_mlp_exp/nanoGPT")
import torch
from model import GPTConfig, GPT


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


configs = [
    (
        "Vanilla Standard (6L 12H)",
        GPTConfig(
            n_layer=6,
            n_head=12,
            n_embd=768,
            vocab_size=50304,
            block_size=1024,
            bias=False,
            use_no_mlp=False,
        ),
    ),
    (
        "No-MLP Standard (6L 12H)",
        GPTConfig(
            n_layer=6,
            n_head=12,
            n_embd=768,
            vocab_size=50304,
            block_size=1024,
            bias=False,
            use_no_mlp=True,
            value_dim=3072,
        ),
    ),
    (
        "Vanilla Large (6L 16H)",
        GPTConfig(
            n_layer=6,
            n_head=16,
            n_embd=2048,
            vocab_size=50304,
            block_size=1024,
            bias=False,
            use_no_mlp=False,
        ),
    ),
    (
        "No-MLP Large (6L 16H)",
        GPTConfig(
            n_layer=6,
            n_head=16,
            n_embd=2048,
            vocab_size=50304,
            block_size=1024,
            bias=False,
            use_no_mlp=True,
            value_dim=8192,
        ),
    ),
    (
        "Vanilla 0.5B mlp3 (26L 16H 1280)",
        GPTConfig(
            n_layer=26,
            n_head=16,
            n_embd=1280,
            vocab_size=50304,
            block_size=1024,
            bias=False,
            use_no_mlp=False,
            mlp_ratio=3.0,
        ),
    ),
    (
        "No-MLP 0.5B (26L 16H 1280)",
        GPTConfig(
            n_layer=26,
            n_head=16,
            n_embd=1280,
            vocab_size=50304,
            block_size=1024,
            bias=False,
            use_no_mlp=True,
            value_dim=5120,
        ),
    ),
]

for name, config in configs:
    model = GPT(config)
    params = count_parameters(model)
    print(f"{name}: {params / 1e6:.2f}M parameters")
