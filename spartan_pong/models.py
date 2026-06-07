from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from spartan_pong.config import ModelConfig


def make_mlp(dim: int, hidden_dim: int, layers: int) -> nn.Sequential:
    # Create an MLP with a given config
    parts: list[nn.Module] = []
    in_dim = dim
    for _ in range(max(1, layers - 1)):
        parts.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
        in_dim = hidden_dim
    parts.append(nn.Linear(in_dim, dim))
    return nn.Sequential(*parts)


def _gumbel_sigmoid(logits: Tensor, temperature: float, hard: bool) -> Tensor:
    # Sample from a Gumbel-Sigmoid distribution for differentiable Bernoulli sampling.
    if not hard:
        return torch.sigmoid(logits)
    # Clamp uniform noise to avoid NaNs from log(0) and log(1).
    uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
    # Gumbel noise is -log(-log(U)) where U ~ Uniform(0, 1). The difference of two Gumbels is log(U) - log(1-U).
    noise = torch.log(uniform) - torch.log1p(-uniform)
    # Add the noise and apply temperature scaling
    soft = torch.sigmoid((logits + noise) / temperature)
    hard_mask = (soft > 0.5).to(soft.dtype)
    # Use hard in the forward pass, but add the soft gradients for the backward pass.
    return hard_mask.detach() - soft.detach() + soft


def path_matrix(adjacencies: Tensor) -> Tensor:
    # Computes the multi-layer path matrix for adjacency tensors of shape [B, L, T, T], where B is batch size, L is number of layers, and T is number of tokens.
    if adjacencies.ndim != 4:
        raise ValueError(f"expected [B, L, T, T], got {tuple(adjacencies.shape)}")
    batch, _, tokens, _ = adjacencies.shape
    eye = torch.eye(tokens, device=adjacencies.device, dtype=adjacencies.dtype).expand(
        batch, tokens, tokens
    )
    path = eye
    for layer in range(adjacencies.shape[1]):
        path = torch.bmm(adjacencies[:, layer] + eye, path)
    return path


class SparseAttentionLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.q = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.k = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.v = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.out = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.mlp = make_mlp(cfg.embed_dim, cfg.mlp_hidden_dim, cfg.mlp_layers)
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)

    def forward(self, x: Tensor, hard: bool = True) -> tuple[Tensor, Tensor, Tensor]:
        # Compute query, key, and value projections
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        # Compute raw attention logits and apply temperature scaling
        edge_logits = torch.matmul(q, k.transpose(-1, -2))
        attention_logits = edge_logits / math.sqrt(q.shape[-1])  # for transformer attention
        probs = torch.sigmoid(
            edge_logits
        )  # for monitoring the Bernoulli probabilities of each edge
        mask = _gumbel_sigmoid(
            edge_logits, self.cfg.hard_temperature, hard=hard
        )  # Sample a hard adjacency mask from the edge logits using Gumbel-Sigmoid
        eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype).unsqueeze(
            0
        )  # ensure self-connections are always present in the mask
        mask = torch.maximum(mask, eye)
        masked_logits = attention_logits.masked_fill(
            mask <= 0.0, -1e9
        )  # mask out the logits of non-edges so they don't contribute to attention weights after softmax
        weights = torch.softmax(
            masked_logits, dim=-1
        )  # compute attention weights from the masked logits

        # Apply attention weights to the value vectors and pass through the output projection and MLP, with residual connections and layer normalization.
        x = self.norm1(x + self.out(torch.matmul(weights, v)))
        x = self.norm2(x + self.mlp(x))
        return x, mask, probs


class SpartanModel(nn.Module):
    model_type = "spartan"

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.token_dim, cfg.embed_dim)
        self.env_tokens = nn.Embedding(cfg.n_envs, cfg.embed_dim)
        self.layers = nn.ModuleList([SparseAttentionLayer(cfg) for _ in range(cfg.layers)])
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim), nn.Linear(cfg.embed_dim, cfg.token_dim)
        )

    def forward(self, x: Tensor, env: Tensor | None = None, hard: bool = True) -> dict[str, Tensor]:
        # Project input tokens to the embedding dimension and concatenate an environment token if provided.
        h = self.in_proj(x)
        if env is None:
            # If no environment indices are provided, use a default index of 0 for all samples.
            env = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        h = torch.cat([h, self.env_tokens(env).unsqueeze(1)], dim=1)
        masks = []
        probs = []
        for layer in self.layers:
            h, mask, prob = layer(h, hard=hard)
            masks.append(mask)
            probs.append(prob)

        # Stack the adjacency masks and probabilities from all layers,
        # compute the path matrix,
        # and produce the final predictions from the output projection. Also compute a sparsity metric based on the path matrix.
        adjacency = torch.stack(masks, dim=1)
        prob_adjacency = torch.stack(probs, dim=1)
        paths = path_matrix(adjacency)
        # Get the prediction (the final 4 object tokens)
        pred = self.out(h[:, : self.cfg.n_objects])
        graph_paths = paths[:, : self.cfg.n_objects]
        sparsity = graph_sparsity_count(graph_paths)
        return {
            "pred": pred,
            "adjacency": adjacency,
            "attention": prob_adjacency,
            "path": paths,
            "sparsity": sparsity,
        }

    @torch.no_grad()
    def graph(
        self,
        x: Tensor,
        env: Tensor | None = None,
        include_env: bool = False,
        threshold: float | None = None,
    ) -> Tensor:
        out = self.forward(x, env=env, hard=False)
        if threshold is None:
            threshold = self.cfg.attention_threshold
        probs = (out["attention"] > threshold).to(x.dtype)
        paths = path_matrix(probs)
        parent_count = self.cfg.n_objects + int(include_env)
        return (paths[:, : self.cfg.n_objects, :parent_count] >= 1.0).to(torch.uint8)


class TransformerLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.q = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.k = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.v = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.out = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.mlp = make_mlp(cfg.embed_dim, cfg.mlp_hidden_dim, cfg.mlp_layers)
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Classic dense Transformer attention layer without sparsity, for the dense baseline model.
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        weights = torch.softmax(logits, dim=-1)
        x = self.norm1(x + self.out(torch.matmul(weights, v)))
        x = self.norm2(x + self.mlp(x))
        return x, weights


class TransformerModel(nn.Module):
    model_type = "transformer"

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.token_dim, cfg.embed_dim)
        self.env_tokens = nn.Embedding(cfg.n_envs, cfg.embed_dim)
        self.layers = nn.ModuleList([TransformerLayer(cfg) for _ in range(cfg.layers)])
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim), nn.Linear(cfg.embed_dim, cfg.token_dim)
        )

    def forward(self, x: Tensor, env: Tensor | None = None, hard: bool = True) -> dict[str, Tensor]:
        del hard  # for api compatibility, but transformer doesn't use hard sampling
        h = self.in_proj(x)
        if env is None:
            env = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        h = torch.cat([h, self.env_tokens(env).unsqueeze(1)], dim=1)
        weights = []
        for layer in self.layers:
            h, attn = layer(h)
            weights.append(attn)
        pred = self.out(h[:, : self.cfg.n_objects])
        return {
            "pred": pred,
            "attention": torch.stack(weights, dim=1),
            "sparsity": pred.new_tensor(0.0),
        }

    @torch.no_grad()
    def graph(
        self,
        x: Tensor,
        env: Tensor | None = None,
        include_env: bool = False,
        threshold: float | None = None,
    ) -> Tensor:
        out = self.forward(x, env=env, hard=False)
        # Dense Transformer attention is softmax-normalised, so a 0.5 cutoff
        # commonly extracts no edges. Above-uniform attention is the natural
        # dense-baseline analogue of SPARTAN's Bernoulli edge threshold.
        if threshold is None:
            threshold = 1.0 / out["attention"].shape[-1]
        adj = (out["attention"] > threshold).to(x.dtype)
        paths = path_matrix(adj)
        parent_count = self.cfg.n_objects + int(include_env)
        return (paths[:, : self.cfg.n_objects, :parent_count] >= 1.0).to(torch.uint8)


def build_model(model_type: str, cfg: ModelConfig) -> nn.Module:
    if model_type == "spartan":
        return SpartanModel(cfg)
    if model_type == "transformer":
        return TransformerModel(cfg)
    raise ValueError(f"unknown model type: {model_type}")


def graph_sparsity_count(graph_paths: Tensor, n_objects: int | None = None) -> Tensor:
    """Return paper-style |A_bar|: per-sample non-self path count averaged over batch."""
    if graph_paths.ndim != 3:
        raise ValueError(f"expected [B, O, P], got {tuple(graph_paths.shape)}")
    if n_objects is None:
        n_objects = graph_paths.shape[1]
    path_exists = torch.clamp(graph_paths[:, :n_objects], 0.0, 1.0)
    eye = torch.eye(path_exists.shape[-1], device=graph_paths.device, dtype=graph_paths.dtype)[
        :n_objects
    ]
    non_self_paths = path_exists * (1.0 - eye.unsqueeze(0))
    return non_self_paths.sum(dim=(1, 2)).mean()


def checkpoint_payload(
    model: nn.Module, cfg: ModelConfig, model_type: str, extra: dict[str, Any]
) -> dict[str, Any]:
    return {
        "model_type": model_type,
        "model_config": cfg.__dict__,
        "state_dict": model.state_dict(),
        **extra,
    }


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> nn.Module:
    payload = torch.load(path, map_location=map_location)
    cfg = ModelConfig(**payload["model_config"])
    model = build_model(payload["model_type"], cfg)
    model.load_state_dict(payload["state_dict"])
    return model
