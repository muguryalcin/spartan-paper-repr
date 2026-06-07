"""Check VAE reconstruction quality of Pong score-object slots."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add repo root to path so spartan_pong imports work when run as script
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from spartan_pong.config import N_OBJECTS  # noqa: E402
from spartan_pong.data import load_npz  # noqa: E402
from spartan_pong.vae import ObjectVAE, _write_png, object_inputs  # noqa: E402

# Score object is index 3 (left_paddle=0, right_paddle=1, ball=2, score=3)
SCORE_OBJ = 3
IMAGE_SIZE = 32
GAP = 4


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find Pong frames with non-zero score and check VAE score-object reconstruction."
    )
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Load the pixel dataset and separate frames by whether a non-zero score is present.
    dev = torch.device(args.device)
    data = load_npz(args.data)
    images = data["image"]
    masks = data["mask"]
    n_total = len(images)

    score_values = _score_values(data, masks)
    scoring = np.where(score_values > 0)[0]
    nonscoring = np.where(score_values <= 0)[0]

    print(f"total frames: {n_total}")
    print(f"non-zero score frames: {len(scoring)}")
    print(f"zero-score frames: {len(nonscoring)}")

    # Sample up to max_samples total, preferring scoring frames if available
    n_score = min(args.max_samples, len(scoring))
    n_zero = min(max(0, args.max_samples - n_score), len(nonscoring))

    if n_score == 0:
        print("No scoring frames found. Only showing zero-score frames for comparison.")
        n_zero = min(args.max_samples, len(nonscoring))

    selected: list[int] = []
    rng = np.random.default_rng(42)
    if n_score > 0:
        selected.extend(rng.choice(scoring, size=n_score, replace=False).tolist())
    if n_zero > 0:
        selected.extend(rng.choice(nonscoring, size=n_zero, replace=False).tolist())

    print(f"selected {len(selected)} frames ({n_score} scoring, {n_zero} zero-score)")

    # Load the trained VAE model from the checkpoint
    ckpt = torch.load(args.vae_checkpoint, map_location=dev)
    vae = ObjectVAE(latent_dim=int(ckpt["latent_dim"])).to(dev)
    vae.load_state_dict(ckpt["state_dict"])
    vae.eval()

    # Convert selected frames into object slot inputs and run VAE reconstruction
    selected_images = images[selected]
    selected_masks = masks[selected]

    all_objects = object_inputs(selected_images, selected_masks)
    n_frames = len(selected)
    assert len(all_objects) == n_frames * N_OBJECTS

    with torch.no_grad():
        x_in = torch.as_tensor(all_objects, dtype=torch.float32, device=dev)
        recon = vae(x_in)["recon"].cpu().numpy()

    # Compute masked reconstruction: only penalize pixels within each object's mask
    targets = all_objects[:, :3]
    masks_flat = all_objects[:, 3:4]
    masked_recons = recon * masks_flat

    # Compute per-frame, per-object MSE and L2 to compare reconstruction quality
    OBJECT_NAMES = ["left_paddle", "right_paddle", "ball", "score"]

    print("\nPer-frame reconstruction metrics:")
    print(
        f"{'frame':>5} {'score':>6} {'left_paddle_mse':>16} {'right_paddle_mse':>16} {'ball_mse':>12} {'score_mse':>12} {'score_l2':>10}"
    )

    per_obj_mse: dict[str, list[float]] = {name: [] for name in OBJECT_NAMES}
    per_obj_l2: dict[str, list[float]] = {name: [] for name in OBJECT_NAMES}

    for frame_idx in range(n_frames):
        score_value = score_values[selected[frame_idx]]
        row = [f"{frame_idx:>5}", f"{score_value:>6.1f}"]
        for obj_idx, obj_name in enumerate(OBJECT_NAMES):
            flat_idx = frame_idx * N_OBJECTS + obj_idx
            orig = targets[flat_idx]
            recon_obj = masked_recons[flat_idx]
            mse = float(np.mean((orig - recon_obj) ** 2))
            l2 = float(np.linalg.norm(orig - recon_obj))
            per_obj_mse[obj_name].append(mse)
            per_obj_l2[obj_name].append(l2)
            if obj_name == "score":
                row.append(f"{mse:>12.2e}")
                row.append(f"{l2:>10.4f}")
            else:
                row.append(f"{mse:>16.2e}")
        print(" ".join(row))

    # Print aggregate MSE per object across all selected frames
    print("\nMean MSE across all frames:")
    for obj_name in OBJECT_NAMES:
        mean_mse = float(np.mean(per_obj_mse[obj_name]))
        print(f"  {obj_name}: {mean_mse:.2e}")

    # Assess score reconstruction quality relative to other objects
    score_mse = float(np.mean(per_obj_mse["score"]))
    paddle_mse = float(
        np.mean([np.mean(per_obj_mse["left_paddle"]), np.mean(per_obj_mse["right_paddle"])])
    )
    ball_mse = float(np.mean(per_obj_mse["ball"]))

    ratio = score_mse / max(paddle_mse, ball_mse)
    if ratio < 2.0:
        quality = "GOOD"
    elif ratio < 10.0:
        quality = "OK"
    else:
        quality = "POOR"

    print(f"\nScore reconstruction quality: {quality}")
    print(f"Score MSE / max(paddle, ball) MSE = {ratio:.2f}")
    if quality == "POOR":
        print("WARNING: VAE may have failed to learn score object properly.")
    elif quality == "OK":
        print("Score reconstruction is acceptable but weaker than other objects.")
    else:
        print("Score reconstruction is comparable to other objects.")

    # Build a visual contact sheet showing original vs reconstructed score masks side-by-side
    targets_nhwc = np.transpose(targets, (0, 2, 3, 1))
    recons_nhwc = np.transpose(masked_recons, (0, 2, 3, 1))

    score_targets: list[np.ndarray] = []
    score_recons: list[np.ndarray] = []
    for frame_idx in range(n_frames):
        flat_idx = frame_idx * N_OBJECTS + SCORE_OBJ
        score_targets.append(targets_nhwc[flat_idx])
        score_recons.append(recons_nhwc[flat_idx])

    # Interleave targets and reconstructions: [t0, r0, t1, r1, ...]
    tiles: list[np.ndarray] = []
    for i in range(len(selected)):
        tiles.append(score_targets[i])
        tiles.append(score_recons[i])

    # Build a 1-row contact sheet manually instead of using _contact_sheet (single row with many columns)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 1
    n_cols = len(tiles)
    sheet = np.ones(
        (n_rows * IMAGE_SIZE + (n_rows + 1) * GAP, n_cols * IMAGE_SIZE + (n_cols + 1) * GAP, 3),
        dtype=np.float32,
    )
    for idx, tile in enumerate(tiles):
        row = idx // n_cols
        col = idx % n_cols
        y0 = GAP + row * (IMAGE_SIZE + GAP)
        x0 = GAP + col * (IMAGE_SIZE + GAP)
        sheet[y0 : y0 + IMAGE_SIZE, x0 : x0 + IMAGE_SIZE] = tile

    _write_png(out_path, sheet)
    print(f"wrote {out_path}")
    print("Each pair: left=original score mask, right=reconstructed score mask")
    print(f"First {n_score} pairs have score>0; last {n_zero} pairs have score=0 for comparison.")


def _score_values(data: dict[str, np.ndarray], masks: np.ndarray) -> np.ndarray:
    if "score_left" in data and "score_right" in data:
        return data["score_left"].astype(np.float32) + data["score_right"].astype(np.float32)
    # Older datasets did not store score metadata. Treat visible score glyph slots
    # as available score examples, but their exact numeric value is unknown.
    score_visible = masks[:, SCORE_OBJ].reshape(len(masks), -1).sum(axis=1) > 0.0
    return score_visible.astype(np.float32)


if __name__ == "__main__":
    main()
