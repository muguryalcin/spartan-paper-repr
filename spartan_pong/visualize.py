from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from spartan_pong.config import N_OBJECTS, TOKEN_DIM
from spartan_pong.data import load_npz
from spartan_pong.models import load_checkpoint
from spartan_pong.vae import ObjectVAE

# Display labels for the two model types in visualizations
MODEL_LABELS = {
    "transformer": "Transformer pred",
    "spartan": "SPARTAN pred",
}


def visualize_outputs(
    pixels_path: str | Path,
    tokens_path: str | Path,
    vae_checkpoint: str | Path,
    transformer_checkpoint: str | Path,
    spartan_checkpoint: str | Path,
    out_dir: str | Path,
    num_samples: int = 8,
    episodes: int = 1,
    rollout_steps: int = 16,
    fps: int = 4,
    device: str = "cpu",
) -> dict[str, Any]:
    # Load pixel and token datasets and validate they are compatible
    pixels = load_npz(pixels_path)
    tokens = load_npz(tokens_path)
    _validate_inputs(pixels, tokens)

    # Load the VAE decoder and both dynamics models onto the target device
    dev = torch.device(device)
    vae = _load_vae(vae_checkpoint, dev)
    models = {
        "transformer": load_checkpoint(str(transformer_checkpoint), map_location=dev).to(dev),
        "spartan": load_checkpoint(str(spartan_checkpoint), map_location=dev).to(dev),
    }

    # Create the output directory
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Generate a contact sheet comparing one-step predictions from both models against the target frames
    sample_indices = list(range(min(num_samples, len(tokens["x"]))))
    sheet_path = out / "contact_sheet.png"
    _write_contact_sheet(sheet_path, pixels, tokens, vae, models, sample_indices, dev)

    # Generate rollout GIFs for selected episodes showing autoregressive multi-step predictions
    selected_episodes = _select_episode_indices(tokens, episodes, rollout_steps)
    gif_paths = []
    for episode_indices in selected_episodes:
        env_id = int(tokens["env"][episode_indices[0]])
        episode_id = int(tokens["episode"][episode_indices[0]])
        gif_path = out / f"rollout_env{env_id}_episode{episode_id}.gif"
        _write_rollout_gif(gif_path, pixels, tokens, vae, models, episode_indices, fps, dev)
        gif_paths.append(str(gif_path))

    # Save metadata describing all generated outputs
    metadata = {
        "pixels": str(pixels_path),
        "tokens": str(tokens_path),
        "vae_checkpoint": str(vae_checkpoint),
        "transformer_checkpoint": str(transformer_checkpoint),
        "spartan_checkpoint": str(spartan_checkpoint),
        "num_samples": len(sample_indices),
        "sample_indices": sample_indices,
        "episodes": [
            {
                "env": int(tokens["env"][indices[0]]),
                "episode": int(tokens["episode"][indices[0]]),
                "indices": indices.tolist(),
            }
            for indices in selected_episodes
        ],
        "rollout_steps": rollout_steps,
        "fps": fps,
        "device": device,
        "outputs": {
            "contact_sheet": str(sheet_path),
            "rollout_gifs": gif_paths,
        },
    }
    metadata_path = out / "visualization_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def _validate_inputs(pixels: dict[str, np.ndarray], tokens: dict[str, np.ndarray]) -> None:
    # Check that both datasets have the required fields and are consistent with each other
    pixel_required = {"image", "next_image", "mask", "next_mask", "env", "episode", "t"}
    token_required = {"x", "y", "env", "episode", "t"}
    missing_pixels = sorted(pixel_required.difference(pixels))
    missing_tokens = sorted(token_required.difference(tokens))
    if missing_pixels:
        raise ValueError(f"pixel dataset missing fields: {missing_pixels}")
    if missing_tokens:
        raise ValueError(f"token dataset missing fields: {missing_tokens}")
    if len(pixels["image"]) != len(tokens["x"]):
        raise ValueError(
            f"pixel/token length mismatch: {len(pixels['image'])} vs {len(tokens['x'])}"
        )
    # Verify that metadata arrays (env, episode, time) match between the two datasets
    for key in ("env", "episode", "t"):
        if not np.array_equal(pixels[key], tokens[key]):
            raise ValueError(f"pixel/token metadata mismatch for {key}")
    # Validate token shape is [N, N_OBJECTS, TOKEN_DIM]
    if tokens["x"].shape[1:] != (N_OBJECTS, TOKEN_DIM):
        raise ValueError(
            f"expected token x shape [N,{N_OBJECTS},{TOKEN_DIM}], got {tokens['x'].shape}"
        )


def _load_vae(checkpoint: str | Path, device: torch.device) -> ObjectVAE:
    # Load the VAE decoder from a saved checkpoint, extract latent dim from the payload
    payload = torch.load(checkpoint, map_location=device)
    model = ObjectVAE(latent_dim=int(payload["latent_dim"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def _predict_next(
    model: nn.Module,
    x_np: np.ndarray,
    env_np: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    # Run a one-step prediction: given current tokens, predict the next tokens
    model.eval()
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    env = torch.as_tensor(env_np, dtype=torch.long, device=device)
    return model(x, env=env, hard=False)["pred"].cpu().numpy()


@torch.no_grad()
def _decode_tokens(vae: ObjectVAE, tokens_np: np.ndarray, device: torch.device) -> np.ndarray:
    # Decode object tokens back to pixel-space images using the VAE decoder.
    # Transpose from [objects, batch, token_dim] to flat [objects*batch, token_dim],
    # decode through VAE, reshape back to [batch, objects, 32, 32, 3], and max-pool over objects.
    n = len(tokens_np)
    flat = tokens_np.transpose(1, 0, 2).reshape(N_OBJECTS * n, TOKEN_DIM)
    z = torch.as_tensor(flat, dtype=torch.float32, device=device)
    decoded = vae.decode(z).cpu().numpy()
    slots = decoded.reshape(N_OBJECTS, n, 3, 32, 32).transpose(1, 0, 3, 4, 2)
    return np.clip(slots.max(axis=1), 0.0, 1.0)


def _one_step_prediction_frames(
    tokens: dict[str, np.ndarray],
    vae: ObjectVAE,
    models: dict[str, nn.Module],
    indices: np.ndarray | list[int],
    device: torch.device,
) -> dict[str, np.ndarray]:
    # For each model, run one-step predictions at the given indices and decode to images
    idx = np.asarray(indices, dtype=np.int64)
    frames = {}
    for name, model in models.items():
        pred_tokens = _predict_next(model, tokens["x"][idx], tokens["env"][idx], device)
        frames[name] = _decode_tokens(vae, pred_tokens, device)
    return frames


def _write_contact_sheet(
    path: Path,
    pixels: dict[str, np.ndarray],
    tokens: dict[str, np.ndarray],
    vae: ObjectVAE,
    models: dict[str, nn.Module],
    sample_indices: list[int],
    device: torch.device,
) -> None:
    # Arrange a grid of current frame, target next frame, and both model predictions into a labeled comparison image
    if not sample_indices:
        raise ValueError("num_samples must select at least one sample")
    idx = np.asarray(sample_indices, dtype=np.int64)
    pred_frames = _one_step_prediction_frames(tokens, vae, models, idx, device)
    rows = []
    for pos, sample_idx in enumerate(idx):
        rows.append(
            [
                pixels["image"][sample_idx],
                pixels["next_image"][sample_idx],
                pred_frames["transformer"][pos],
                pred_frames["spartan"][pos],
            ]
        )
    image = _labeled_grid(
        rows,
        column_labels=[
            "current",
            "target next",
            MODEL_LABELS["transformer"],
            MODEL_LABELS["spartan"],
        ],
    )
    image.save(path)


def _select_episode_indices(
    tokens: dict[str, np.ndarray], max_episodes: int, rollout_steps: int
) -> list[np.ndarray]:
    # Find up to max_episodes complete episodes in the dataset and return the first rollout_steps time indices per episode
    if max_episodes <= 0:
        raise ValueError("episodes must be positive")
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    envs = tokens["env"].astype(int)
    episode_ids = tokens["episode"].astype(int)
    t_values = tokens["t"].astype(int)
    selected = []
    for env_id, episode_id in sorted(set(zip(envs.tolist(), episode_ids.tolist(), strict=True))):
        idx = np.where((envs == env_id) & (episode_ids == episode_id))[0]
        idx = idx[np.argsort(t_values[idx])][:rollout_steps]
        if len(idx) > 0:
            selected.append(idx)
        if len(selected) >= max_episodes:
            break
    if not selected:
        raise ValueError("no episodes available for rollout visualization")
    return selected


def _rollout_prediction_frames(
    tokens: dict[str, np.ndarray],
    vae: ObjectVAE,
    model: nn.Module,
    episode_indices: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    # Run an autoregressive rollout through the episode: feed the model's own predictions back as input
    model.eval()
    env = torch.full((1,), int(tokens["env"][episode_indices[0]]), dtype=torch.long, device=device)
    current = torch.as_tensor(
        tokens["x"][episode_indices[0] : episode_indices[0] + 1],
        dtype=torch.float32,
        device=device,
    )
    preds = []
    with torch.no_grad():
        for _ in episode_indices:
            current = model(current, env=env, hard=False)["pred"]
            preds.append(current.cpu().numpy()[0])
    return _decode_tokens(vae, np.asarray(preds, dtype=np.float32), device)


def _write_rollout_gif(
    path: Path,
    pixels: dict[str, np.ndarray],
    tokens: dict[str, np.ndarray],
    vae: ObjectVAE,
    models: dict[str, nn.Module],
    episode_indices: np.ndarray,
    fps: int,
    device: torch.device,
) -> None:
    # Generate an animated GIF comparing ground-truth vs model predictions for one episode
    if fps <= 0:
        raise ValueError("fps must be positive")
    transformer_frames = _rollout_prediction_frames(
        tokens, vae, models["transformer"], episode_indices, device
    )
    spartan_frames = _rollout_prediction_frames(
        tokens, vae, models["spartan"], episode_indices, device
    )
    frames = []
    for pos, sample_idx in enumerate(episode_indices):
        frame = _labeled_grid(
            [
                [
                    pixels["image"][sample_idx],
                    pixels["next_image"][sample_idx],
                    transformer_frames[pos],
                    spartan_frames[pos],
                ]
            ],
            column_labels=[
                "current",
                "target next",
                MODEL_LABELS["transformer"],
                MODEL_LABELS["spartan"],
            ],
        )
        frames.append(frame)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=max(1, int(1000 / fps)),
        loop=0,
    )


def _to_tile(image: np.ndarray, scale: int) -> Image.Image:
    # Convert a float [0,1] numpy image to a PIL Image scaled up by NEAREST-neighbor (pixel-art style)
    arr = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB").resize(
        (arr.shape[1] * scale, arr.shape[0] * scale), Image.Resampling.NEAREST
    )


def _labeled_grid(
    rows: list[list[np.ndarray]],
    column_labels: list[str],
    scale: int = 6,
    label_height: int = 18,
    gap: int = 4,
) -> Image.Image:
    # Build a labeled grid of image tiles with column headers and upscaled tiles for visual comparison
    if not rows or not rows[0]:
        raise ValueError("grid requires at least one row and one column")
    cols = len(rows[0])
    tile_h, tile_w = rows[0][0].shape[:2]
    tile_size = (tile_w * scale, tile_h * scale)
    width = cols * tile_size[0] + (cols + 1) * gap
    height = label_height + len(rows) * tile_size[1] + (len(rows) + 1) * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    # Draw the column labels at the top
    for col, label in enumerate(column_labels):
        x = gap + col * (tile_size[0] + gap)
        draw.text((x, 3), label, fill=(0, 0, 0), font=font)
    # Paste each tile into the grid, upscaled by the scale factor
    for row_idx, row in enumerate(rows):
        if len(row) != cols:
            raise ValueError("all grid rows must have the same number of columns")
        y = label_height + gap + row_idx * (tile_size[1] + gap)
        for col_idx, image in enumerate(row):
            x = gap + col_idx * (tile_size[0] + gap)
            canvas.paste(_to_tile(image, scale), (x, y))
    return canvas
