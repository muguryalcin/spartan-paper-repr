from __future__ import annotations

from dataclasses import dataclass

OBJECTS = ("left_paddle", "right_paddle", "ball", "score")
N_OBJECTS = len(OBJECTS)
TOKEN_DIM = 32
# We use 7 environments for training and 4 for testing
# They are mentioned in the Table 5 of the paper.
SEEN_ENVS = tuple(range(7))
UNSEEN_ENVS = tuple(range(7, 11))


@dataclass(frozen=True)
class ModelConfig:
    token_dim: int = TOKEN_DIM
    embed_dim: int = 512
    layers: int = 3
    mlp_hidden_dim: int = 512
    mlp_layers: int = 3
    n_objects: int = N_OBJECTS
    n_envs: int = 11
    hard_temperature: float = 0.5
    attention_threshold: float = 0.5


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 100_000
    batch_size: int = 256
    lr: float = 5e-5
    seed: int = 0
    eval_every: int = 2_000
    target_loss: float = 1e-3
    lagrangian_alpha: float = 1.001
    lambda_init: float = 20.0
    sparsity_weight: float = 1.0
    device: str = "cpu"
