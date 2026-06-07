from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm, trange

from spartan_pong.config import N_OBJECTS, TOKEN_DIM
from spartan_pong.data import load_npz


class ObjectVAE(nn.Module):
    """Small masked-object VAE for 32x32 Pong object slots."""

    def __init__(self, latent_dim: int = TOKEN_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Flatten(),
        )
        # mean
        self.enc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        # log-variance
        self.enc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
        self.dec_in = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Encode input to latent mean and log-variance. Log-variance is clamped to a reasonable range to avoid numerical issues.
        h = self.encoder(x)
        return self.enc_mu(h), self.enc_logvar(h).clamp(-10.0, 10.0)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        # Reparameterization trick to sample from the latent distribution.
        # During evaluation, we use the mean directly for deterministic output.
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: Tensor) -> Tensor:
        # Decode the latent vector back to the original input space. The output is passed through a sigmoid to ensure it is in the range [0, 1].
        return self.decoder(self.dec_in(z))

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        # Forward pass
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}


def object_inputs(images: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Return [N * objects, 4, 32, 32] masked RGB plus mask channel."""
    if images.ndim != 4 or masks.ndim != 4:
        raise ValueError("expected images [N,H,W,3] and masks [N,O,H,W]")
    rgb = np.transpose(images, (0, 3, 1, 2))
    slots = []
    for obj in range(masks.shape[1]):
        mask = masks[:, obj : obj + 1]
        slots.append(np.concatenate([rgb * mask, mask], axis=1))
    return np.concatenate(slots, axis=0).astype(np.float32)


def _write_png(path: Path, image: np.ndarray) -> None:
    image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image_u8).save(path)


def _contact_sheet(
    tiles: list[np.ndarray], rows: int, cols: int, tile_size: int, gap: int = 1
) -> np.ndarray:
    # Creates a grid from the given tiles.
    height = rows * tile_size + (rows + 1) * gap
    width = cols * tile_size + (cols + 1) * gap
    sheet = np.ones((height, width, 3), dtype=np.float32)
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        y0 = gap + row * (tile_size + gap)
        x0 = gap + col * (tile_size + gap)
        sheet[y0 : y0 + tile_size, x0 : x0 + tile_size] = tile
    return sheet


def _vae_loss(out: dict[str, Tensor], x: Tensor, beta: float) -> tuple[Tensor, Tensor, Tensor]:
    # Compute the VAE loss as the sum of the reconstruction loss (MSE) and the KL divergence, weighted by beta.
    target = x[:, :3]
    mask = x[:, 3:4]
    recon = F.mse_loss(out["recon"] * mask, target * mask, reduction="mean")
    kl = -0.5 * torch.mean(1.0 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
    return recon + beta * kl, recon, kl


def _save_vae_checkpoint(
    checkpoint_path: Path,
    model: ObjectVAE,
    opt: torch.optim.Optimizer,
    metrics: dict[str, float],
    config: dict[str, str | int | float],
    step: int,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "latent_dim": model.latent_dim,
            "metrics": metrics,
            "config": config,
            "step": step,
        },
        checkpoint_path,
    )


def train_vae(
    data_path: str | Path,
    out_dir: str | Path,
    steps: int = 10_000,
    batch_size: int = 256,
    lr: float = 1e-3,
    beta: float = 1e-4,
    seed: int = 0,
    device: str = "cpu",
    resume: bool = False,
) -> dict[str, float]:
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    # load data
    data = load_npz(data_path)
    if "image" not in data or "mask" not in data:
        raise ValueError("VAE training requires a generated pixel dataset")
    # prepare 4-channel masked RGB inputs for each object slot
    inputs = object_inputs(data["image"], data["mask"])
    # build model and optimizer
    dev = torch.device(device)
    model = ObjectVAE().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    metrics: dict[str, float] = {}
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_path / "vae.pt"
    start_step = 0
    config = {
        "data_path": str(data_path),
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "beta": beta,
        "seed": seed,
        "device": device,
        "latent_dim": model.latent_dim,
    }
    # Resume from checkpoint if requested and available
    if resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=dev)
        model.load_state_dict(payload["state_dict"])
        if "optimizer_state" in payload:
            opt.load_state_dict(payload["optimizer_state"])
        metrics = dict(payload.get("metrics", {}))
        start_step = int(payload.get("step", -1)) + 1
        if start_step >= steps:
            return metrics
    # train
    pbar = trange(start_step, steps, desc="train vae")
    for step in pbar:
        # Sample a random batch of object slots and perform a training step
        idx = np.random.randint(0, len(inputs), size=batch_size)
        x = torch.as_tensor(inputs[idx], dtype=torch.float32, device=dev)
        out = model(x)
        loss, recon, kl = _vae_loss(out, x, beta)
        # update modelj
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        # Log metrics and save checkpoint periodically
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            metrics = {
                "loss": float(loss.detach().cpu()),
                "recon": float(recon.detach().cpu()),
                "kl": float(kl.detach().cpu()),
                "step": float(step),
            }
            pbar.set_postfix({k: f"{v:.4g}" for k, v in metrics.items() if k != "step"})
            # save the model
            _save_vae_checkpoint(checkpoint_path, model, opt, metrics, config, step)
    _save_vae_checkpoint(checkpoint_path, model, opt, metrics, config, steps - 1)
    (out_path / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    (out_path / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    print(f"  VAE done — recon={metrics['recon']:.4f}, kl={metrics['kl']:.4f}")
    return metrics


@torch.no_grad()
def _encode_slots(
    model: ObjectVAE, images: np.ndarray, masks: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    # Encodes the object slots for the given images and masks using the VAE
    # The input images and masks are first converted into the appropriate format for the VAE,
    #  then processed in batches to obtain the latent representations (mu) for each object slot.
    inputs = object_inputs(images, masks)
    encoded = []
    model.eval()
    for start in tqdm(range(0, len(inputs), batch_size), desc="Encoding tokens", leave=False):
        x = torch.as_tensor(inputs[start : start + batch_size], dtype=torch.float32, device=device)
        mu, _ = model.encode(x)
        encoded.append(mu.cpu().numpy())
    flat = np.concatenate(encoded, axis=0)
    n = len(images)
    return flat.reshape(N_OBJECTS, n, TOKEN_DIM).transpose(1, 0, 2).astype(np.float32)


def encode_dataset_tokens(
    checkpoint: str | Path,
    data_path: str | Path,
    out_path: str | Path,
    device: str = "cpu",
    batch_size: int = 512,
) -> None:
    # Load the pixel dataset, encode the object slots using the trained VAE, and save the resulting token dataset.
    data = load_npz(data_path)
    required = {"image", "mask", "next_image", "next_mask"}
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"VAE token export requires pixel dataset fields missing: {missing}")
    dev = torch.device(device)
    payload = torch.load(checkpoint, map_location=dev)
    model = ObjectVAE(latent_dim=int(payload["latent_dim"])).to(dev)
    model.load_state_dict(payload["state_dict"])
    out = {
        "x": _encode_slots(model, data["image"], data["mask"], dev, batch_size),
        "y": _encode_slots(model, data["next_image"], data["next_mask"], dev, batch_size),
        "graph": data["graph"],
        "graph_with_env": data["graph_with_env"] if "graph_with_env" in data else data["graph"],
        "env": data["env"],
        "episode": data["episode"],
        "t": data["t"],
        "split": data["split"],
        "token_source": np.asarray("object_vae"),
    }
    for key in ("score_left", "score_right"):
        if key in data:
            out[key] = data[key]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)


@torch.no_grad()
def export_reconstruction_sheet(
    checkpoint: str | Path,
    data_path: str | Path,
    out_dir: str | Path,
    num_samples: int = 8,
    device: str = "cpu",
    batch_size: int = 128,
) -> dict[str, float]:
    # Load the pixel dataset, reconstruct the object slots using the trained VAE,
    # and save a contact sheet comparing the original and reconstructed slots for a sample of frames.
    # Also compute and save reconstruction metrics.
    # Load the pixel dataset and validate required fields exist
    data = load_npz(data_path)
    required = {"image", "mask"}
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(
            f"VAE reconstruction export requires pixel dataset fields missing: {missing}"
        )
    # Clamp the number of samples to the available frames
    sample_count = min(num_samples, len(data["image"]))
    if sample_count <= 0:
        raise ValueError("num_samples must select at least one frame")

    # Load the trained VAE model from the checkpoint
    dev = torch.device(device)
    payload = torch.load(checkpoint, map_location=dev)
    model = ObjectVAE(latent_dim=int(payload["latent_dim"])).to(dev)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    # Convert the first sample_count frames into object slot inputs and reconstruct them through the VAE in batches
    inputs = object_inputs(data["image"][:sample_count], data["mask"][:sample_count])
    recon_chunks = []
    for start in tqdm(range(0, len(inputs), batch_size), desc="Reconstructing", leave=False):
        x = torch.as_tensor(inputs[start : start + batch_size], dtype=torch.float32, device=dev)
        recon_chunks.append(model(x)["recon"].cpu().numpy())
    recons = np.concatenate(recon_chunks, axis=0)

    # Compute the masked reconstruction MSE: only penalize errors within each object's mask region
    targets = inputs[:, :3]
    masks = inputs[:, 3:4]
    masked_recons = recons * masks
    recon_mse = float(np.mean((masked_recons - targets) ** 2))

    # Transpose targets and reconstructions from CHW to HWC for image layout, then build the contact sheet tiles
    # Each row contains one sample: [obj0_in, obj0_recon, obj1_in, obj1_recon, ...]
    targets_nhwc = np.transpose(targets, (0, 2, 3, 1))
    recons_nhwc = np.transpose(masked_recons, (0, 2, 3, 1))
    tiles: list[np.ndarray] = []
    for sample_idx in range(sample_count):
        for obj in range(N_OBJECTS):
            # Compute the flat index in the object-first layout produced by object_inputs
            flat_idx = obj * sample_count + sample_idx
            tiles.append(targets_nhwc[flat_idx])
            tiles.append(recons_nhwc[flat_idx])

    # Assemble the contact sheet grid and save it as a PNG image
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    sheet = _contact_sheet(
        tiles, rows=sample_count, cols=N_OBJECTS * 2, tile_size=targets.shape[-1]
    )
    _write_png(out_path / "reconstructions.png", sheet)

    # Save the reconstruction metrics to a JSON file
    metrics = {
        "num_samples": float(sample_count),
        "num_slots": float(len(inputs)),
        "masked_recon_mse": recon_mse,
    }
    (out_path / "reconstruction_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True)
    )
    return metrics
