from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from spartan_pong.config import N_OBJECTS, SEEN_ENVS, UNSEEN_ENVS

# CITRIS Interventional Pong geometry/mechanics constants. State coordinates are
# represented on the native 32px scale, not normalized to [0, 1].
RESOLUTION = 32
BORDER_SIZE = 2
PADDLE_H = 6.0
PADDLE_W = 2.0
LEFT_X = 5.0
RIGHT_X = 27.0
BALL_R = 1.2
BALL_VEL_MAGN = 2.0
PADDLE_MAX_STEP = 1.5
PADDLE_STEP_NOISE = 0.5
BALL_X_NOISE = 0.2
BALL_Y_NOISE = 0.2
BALL_VEL_DIR_NOISE = 0.1
MAX_POINTS = 5

# Backwards-compatible aliases for callers/tests that imported the old names.
FIELD_W = float(RESOLUTION)
FIELD_H = float(RESOLUTION)

SCORE_OBJ = 3
BALL_DYNAMICS_ENVS = (1, 4, 5, 6, 7, 8, 9, 10)
LEFT_PADDLE_ENVS = (2, 9)
RIGHT_PADDLE_ENVS = (3, 10)


@dataclass
class PongState:
    left_y: float
    right_y: float
    ball_x: float
    ball_y: float
    ball_vx: float
    ball_vy: float
    score_left: float = 0.0
    score_right: float = 0.0


def _settings() -> dict[str, float]:
    settings = {
        "resolution": float(RESOLUTION),
        "border_size": float(BORDER_SIZE),
        "paddle_height": PADDLE_H,
        "paddle_width": PADDLE_W,
        "paddle_left_x": LEFT_X,
        "paddle_right_x": RIGHT_X,
        "paddle_max_step": PADDLE_MAX_STEP,
        "paddle_step_noise": PADDLE_STEP_NOISE,
        "ball_radius": BALL_R,
        "ball_x_noise": BALL_X_NOISE,
        "ball_y_noise": BALL_Y_NOISE,
        "ball_vel_dir_noise": BALL_VEL_DIR_NOISE,
        "max_points": float(MAX_POINTS),
        "center_point_x": RESOLUTION / 2.0,
        "center_point_y": RESOLUTION / 2.0,
    }
    settings["paddle_left_y_min"] = float(BORDER_SIZE)
    settings["paddle_left_y_max"] = float(RESOLUTION - BORDER_SIZE)
    settings["paddle_right_y_min"] = settings["paddle_left_y_min"]
    settings["paddle_right_y_max"] = settings["paddle_left_y_max"]
    settings["ball_y_min"] = BORDER_SIZE + BALL_R
    settings["ball_y_max"] = RESOLUTION - BORDER_SIZE - BALL_R
    settings["ball_x_min"] = float(BORDER_SIZE)
    settings["ball_x_max"] = float(RESOLUTION - BORDER_SIZE)
    settings["ball_x_min_point"] = BORDER_SIZE + BALL_R
    settings["ball_x_max_point"] = RESOLUTION - BORDER_SIZE - BALL_R
    settings["ball_x_min_sample"] = LEFT_X + PADDLE_W / 2.0 + BALL_R
    settings["ball_x_max_sample"] = RIGHT_X - PADDLE_W / 2.0 - BALL_R
    return settings


SETTINGS = _settings()


def paddle_step(
    paddle_y: float,
    ball_y: float,
    settings: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
    intervention: bool = False,
) -> float:
    """CITRIS paddle transition: track the ball, or random-step under intervention."""
    cfg = SETTINGS if settings is None else settings
    if rng is None:
        rng = np.random.default_rng()
    assert rng is not None
    if not intervention:
        step = min(abs(paddle_y - ball_y), cfg["paddle_max_step"])
        if paddle_y > ball_y:
            step *= -1.0
    else:
        step = cfg["paddle_max_step"]
        if rng.uniform() > 0.5:
            step *= -1.0
    return float(paddle_y + step + rng.normal() * cfg["paddle_step_noise"])


def sample_ball_vel_dir(
    settings: dict[str, float] | None = None, rng: np.random.Generator | None = None
) -> float:
    del settings
    if rng is None:
        rng = np.random.default_rng()
    assert rng is not None
    return float(rng.uniform(0.0, 2.0 * np.pi))


def mod_angle(angle: float) -> float:
    return float(angle % (2.0 * np.pi))


def angle_flip(angle: float, axis: str = "y") -> float:
    """CITRIS angle reflection helper; axis='x' flips vertical velocity."""
    if axis == "x":
        return mod_angle(np.pi - angle)
    if axis == "y":
        return mod_angle(-angle)
    raise ValueError(f"unknown flip axis: {axis}")


def ball_collision(
    paddle_tag: str,
    new_time_step: dict[str, float],
    prev_time_step: dict[str, float],
    settings: dict[str, float] | None = None,
) -> bool:
    """Return whether the new ball position overlaps a paddle from CITRIS geometry."""
    cfg = SETTINGS if settings is None else settings
    paddle_x = cfg[f"{paddle_tag}_x"]
    paddle_y = prev_time_step[f"{paddle_tag}_y"]
    ball_x = new_time_step["ball_x"]
    ball_y = new_time_step["ball_y"]
    if ball_y > paddle_y + cfg["paddle_height"] / 2.0:
        return False
    if ball_y < paddle_y - cfg["paddle_height"] / 2.0:
        return False

    if paddle_tag.endswith("right"):
        ball_x_outer = ball_x + cfg["ball_radius"]
        return bool(
            ball_x < paddle_x + cfg["paddle_width"] / 2.0
            and ball_x_outer > paddle_x - cfg["paddle_width"] / 2.0
        )
    if paddle_tag.endswith("left"):
        ball_x_outer = ball_x - cfg["ball_radius"]
        return bool(
            ball_x > paddle_x - cfg["paddle_width"] / 2.0
            and ball_x_outer < paddle_x + cfg["paddle_width"] / 2.0
        )
    return False


def put_in_boundaries(
    time_step: dict[str, float], settings: dict[str, float] | None = None
) -> dict[str, float]:
    cfg = SETTINGS if settings is None else settings
    for key in tuple(time_step):
        if key == "ball_vel_dir":
            time_step[key] = mod_angle(time_step[key])
        elif f"{key}_min" in cfg and f"{key}_max" in cfg:
            time_step[key] = float(
                np.clip(time_step[key], cfg[f"{key}_min"], cfg[f"{key}_max"])
            )
    return time_step


def _clip_y(y: float, paddle: str = "paddle_left") -> float:
    return float(np.clip(y, SETTINGS[f"{paddle}_y_min"], SETTINGS[f"{paddle}_y_max"]))


def _vel_from_dir(angle: float) -> tuple[float, float]:
    # This matches CITRIS: x velocity uses sin, y velocity uses cos.
    return float(np.sin(angle) * BALL_VEL_MAGN), float(np.cos(angle) * BALL_VEL_MAGN)


def _dir_from_vel(vx: float, vy: float) -> float:
    return mod_angle(float(np.arctan2(vx, vy)))


def _state_to_step(state: PongState) -> dict[str, float]:
    return {
        "paddle_left_y": state.left_y,
        "paddle_right_y": state.right_y,
        "ball_x": state.ball_x,
        "ball_y": state.ball_y,
        "ball_vel_dir": _dir_from_vel(state.ball_vx, state.ball_vy),
        "ball_vel_magn": BALL_VEL_MAGN,
        "score_left": state.score_left,
        "score_right": state.score_right,
    }


def _state_from_step(step: dict[str, float]) -> PongState:
    vx, vy = _vel_from_dir(step["ball_vel_dir"])
    return PongState(
        left_y=float(step["paddle_left_y"]),
        right_y=float(step["paddle_right_y"]),
        ball_x=float(step["ball_x"]),
        ball_y=float(step["ball_y"]),
        ball_vx=vx,
        ball_vy=vy,
        score_left=float(step["score_left"]),
        score_right=float(step["score_right"]),
    )


def _sample_reset(rng: np.random.Generator, score_left: float, score_right: float) -> dict[str, float]:
    step = {
        "paddle_left_y": float(
            rng.uniform(SETTINGS["center_point_y"] * 0.25, SETTINGS["center_point_y"] * 1.75)
        ),
        "paddle_right_y": float(
            rng.uniform(SETTINGS["center_point_y"] * 0.25, SETTINGS["center_point_y"] * 1.75)
        ),
        "ball_x": SETTINGS["center_point_x"],
        "ball_y": SETTINGS["center_point_y"],
        "ball_vel_dir": sample_ball_vel_dir(SETTINGS, rng),
        "ball_vel_magn": BALL_VEL_MAGN,
        "score_left": score_left,
        "score_right": score_right,
    }
    return put_in_boundaries(step, SETTINGS)


def initial_state(rng: np.random.Generator) -> PongState:
    step = _sample_reset(
        rng,
        score_left=float(rng.integers(MAX_POINTS)),
        score_right=float(rng.integers(MAX_POINTS)),
    )
    step["ball_x"] = float(
        rng.uniform(SETTINGS["ball_x_min_sample"], SETTINGS["ball_x_max_sample"])
    )
    step["ball_y"] = float(rng.uniform(SETTINGS["ball_y_min"], SETTINGS["ball_y_max"]))
    return _state_from_step(put_in_boundaries(step, SETTINGS))


def _env_paddle_step(
    side: str,
    paddle_y: float,
    ball_y: float,
    env_id: int,
    rng: np.random.Generator,
) -> float:
    if side == "left" and env_id in LEFT_PADDLE_ENVS:
        return paddle_step(paddle_y, ball_y, SETTINGS, rng, intervention=True)
    if side == "right" and env_id == 3:
        return paddle_step(paddle_y, RESOLUTION - ball_y, SETTINGS, rng)
    if side == "right" and env_id == 10:
        return paddle_step(paddle_y, ball_y, SETTINGS, rng, intervention=True)
    return paddle_step(paddle_y, ball_y, SETTINGS, rng)


def _apply_ball_env_dynamics(
    env_id: int, step: dict[str, float], hit_left: bool, hit_right: bool
) -> None:
    if env_id in (1, 9):
        step["ball_vel_dir"] += 0.12 * np.sin(8.0 * np.pi * step["ball_x"] / RESOLUTION)
    if env_id in (6, 7, 10) and step["ball_x"] < SETTINGS["center_point_x"]:
        step["ball_vel_dir"] = _dir_from_vel(
            *(
                np.array(_vel_from_dir(step["ball_vel_dir"]))
                + np.array([0.0, 0.35], dtype=np.float64)
            )
        )
    if env_id in (4, 8) and 0.38 * RESOLUTION <= step["ball_x"] <= 0.62 * RESOLUTION:
        step["ball_vel_dir"] *= 0.96
    if env_id in (5, 7) or (env_id == 8 and hit_left):
        if hit_left:
            step["ball_vel_dir"] -= 0.18
        elif hit_right:
            step["ball_vel_dir"] += 0.18


def step_state(
    state: PongState, env_id: int, rng: np.random.Generator | None = None
) -> tuple[PongState, np.ndarray]:
    """
    Advance one transition and return the next state plus the local object graph.

    Graph convention is row=child at t+1, column=parent at t.
    """
    if rng is None:
        rng = np.random.default_rng()
    assert rng is not None
    graph = np.zeros((N_OBJECTS, N_OBJECTS), dtype=np.uint8)
    graph[0, 2] = 1  # ball -> left paddle
    graph[1, 2] = 1  # ball -> right paddle

    prev = _state_to_step(state)
    left_y = _clip_y(
        _env_paddle_step("left", state.left_y, state.ball_y, env_id, rng), "paddle_left"
    )
    right_y = _clip_y(
        _env_paddle_step("right", state.right_y, state.ball_y, env_id, rng),
        "paddle_right",
    )

    vx, vy = _vel_from_dir(prev["ball_vel_dir"])
    new = {
        "paddle_left_y": left_y,
        "paddle_right_y": right_y,
        "ball_x": state.ball_x + vx,
        "score_left": state.score_left,
        "score_right": state.score_right,
    }

    point_left = False
    point_right = False
    if new["ball_x"] < SETTINGS["ball_x_min_point"]:
        point_right = True
    elif new["ball_x"] > SETTINGS["ball_x_max_point"]:
        point_left = True

    if point_left or point_right:
        graph[SCORE_OBJ, 2] = 1  # ball -> score
        score_left = state.score_left + float(point_left)
        score_right = state.score_right + float(point_right)
        if max(score_left, score_right) >= MAX_POINTS:
            score_left = 0.0
            score_right = 0.0
        new = _sample_reset(rng, score_left=score_left, score_right=score_right)
    else:
        new["ball_y"] = state.ball_y + vy
        new["ball_vel_dir"] = prev["ball_vel_dir"]
        if new["ball_y"] > SETTINGS["ball_y_max"]:
            new["ball_y"] = SETTINGS["ball_y_max"] - (new["ball_y"] - SETTINGS["ball_y_max"])
            new["ball_vel_dir"] = angle_flip(new["ball_vel_dir"], axis="x")
        elif new["ball_y"] < SETTINGS["ball_y_min"]:
            new["ball_y"] = SETTINGS["ball_y_min"] - (new["ball_y"] - SETTINGS["ball_y_min"])
            new["ball_vel_dir"] = angle_flip(new["ball_vel_dir"], axis="x")

        collision_step = {
            "ball_x": new["ball_x"],
            "ball_y": new["ball_y"],
        }
        hit_left = ball_collision("paddle_left", collision_step, prev, SETTINGS)
        hit_right = ball_collision("paddle_right", collision_step, prev, SETTINGS)
        if hit_left:
            graph[2, 0] = 1  # left paddle -> ball
            new["ball_x"] = (LEFT_X + PADDLE_W / 2.0) * 2.0 - (new["ball_x"] - BALL_R * 2.0)
            new["ball_vel_dir"] = angle_flip(new["ball_vel_dir"], axis="y")
        elif hit_right:
            graph[2, 1] = 1  # right paddle -> ball
            new["ball_x"] = (RIGHT_X - PADDLE_W / 2.0) * 2.0 - (new["ball_x"] + BALL_R * 2.0)
            new["ball_vel_dir"] = angle_flip(new["ball_vel_dir"], axis="y")
        _apply_ball_env_dynamics(env_id, new, hit_left=hit_left, hit_right=hit_right)
        new["ball_x"] += float(rng.normal() * BALL_X_NOISE)
        new["ball_y"] += float(rng.normal() * BALL_Y_NOISE)
        new["ball_vel_dir"] += float(rng.normal() * BALL_VEL_DIR_NOISE)
        new["ball_vel_magn"] = BALL_VEL_MAGN

    next_state = _state_from_step(put_in_boundaries(new, SETTINGS))
    return next_state, graph


def intervention_edges(env_id: int) -> np.ndarray:
    """Return object children whose dynamics are directly changed by the env token."""
    edges = np.zeros(N_OBJECTS, dtype=np.uint8)
    if env_id in LEFT_PADDLE_ENVS:
        edges[0] = 1
    if env_id in RIGHT_PADDLE_ENVS:
        edges[1] = 1
    if env_id in BALL_DYNAMICS_ENVS:
        edges[2] = 1
    return edges


def graph_with_intervention(graph: np.ndarray, env_id: int) -> np.ndarray:
    full = np.zeros((N_OBJECTS, N_OBJECTS + 1), dtype=np.uint8)
    full[:, :N_OBJECTS] = graph
    full[:, N_OBJECTS] = intervention_edges(env_id)
    return full


def _pixel_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size]
    return xx.astype(np.float32) + 0.5, yy.astype(np.float32) + 0.5


def _draw_rect(
    image: np.ndarray,
    masks: np.ndarray,
    obj: int,
    cx: float,
    cy: float,
    width: float,
    height: float,
    color: tuple[float, float, float],
) -> None:
    size = image.shape[0]
    x0 = max(0, int(np.floor(cx - width / 2.0)))
    x1 = min(size, int(np.ceil(cx + width / 2.0)))
    y0 = max(0, int(np.floor(cy - height / 2.0)))
    y1 = min(size, int(np.ceil(cy + height / 2.0)))
    if x0 >= x1 or y0 >= y1:
        return
    image[y0:y1, x0:x1] = color
    masks[obj, y0:y1, x0:x1] = 1.0


def _draw_circle(
    image: np.ndarray,
    masks: np.ndarray,
    obj: int,
    cx: float,
    cy: float,
    radius: float,
    color: tuple[float, float, float],
) -> None:
    xx, yy = _pixel_grid(image.shape[0])
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    image[circle] = color
    masks[obj, circle] = 1.0


_DIGIT_SEGMENTS = {
    0: ((0, 0, 3, 1), (0, 0, 1, 5), (2, 0, 1, 5), (0, 4, 3, 1)),
    1: ((1, 0, 1, 5),),
    2: ((0, 0, 3, 1), (0, 2, 3, 1), (0, 4, 3, 1), (2, 0, 1, 3), (0, 2, 1, 3)),
    3: ((2, 0, 1, 5), (0, 0, 3, 1), (0, 2, 3, 1), (0, 4, 3, 1)),
    4: ((2, 0, 1, 5), (0, 2, 3, 1), (0, 2, 1, 3)),
    5: ((0, 0, 3, 1), (0, 2, 3, 1), (0, 4, 3, 1), (2, 2, 1, 3), (0, 0, 1, 3)),
}


def _draw_score(image: np.ndarray, masks: np.ndarray, left: int, right: int) -> None:
    color = (0.55, 0.55, 0.55)

    def segment(x: int, y: int, width: int, height: int) -> None:
        image[y : y + height, x : x + width] = color
        masks[SCORE_OBJ, y : y + height, x : x + width] = 1.0

    def digit(x: int, y: int, value: int) -> None:
        for dx, dy, width, height in _DIGIT_SEGMENTS[int(value) % 6]:
            segment(x + dx, y + dy, width, height)

    digit(11, 2, left)
    segment(16, 3, 1, 1)
    segment(16, 5, 1, 1)
    digit(19, 2, right)


def render(
    state: PongState, size: int = RESOLUTION, include_ball_trace: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Render a 32x32 RGB frame plus object masks in SPARTAN object order."""
    if size != RESOLUTION:
        raise ValueError(f"Interventional Pong renderer is fixed at {RESOLUTION}px")
    image = np.zeros((size, size, 3), dtype=np.float32)
    masks = np.zeros((N_OBJECTS, size, size), dtype=np.float32)

    _draw_score(image, masks, int(state.score_left), int(state.score_right))
    _draw_rect(image, masks, 0, LEFT_X, state.left_y, PADDLE_W, PADDLE_H, (0.1, 0.35, 1.0))
    _draw_rect(image, masks, 1, RIGHT_X, state.right_y, PADDLE_W, PADDLE_H, (0.1, 0.85, 0.2))
    if include_ball_trace:
        prev_x = float(np.clip(state.ball_x - state.ball_vx, BALL_R, RESOLUTION - BALL_R))
        prev_y = float(np.clip(state.ball_y - state.ball_vy, BALL_R, RESOLUTION - BALL_R))
        _draw_circle(image, masks, 2, prev_x, prev_y, BALL_R, (0.15, 0.55, 1.0))
    _draw_circle(image, masks, 2, state.ball_x, state.ball_y, BALL_R, (1.0, 0.15, 0.1))
    return image, masks


def generate_dataset(
    out: str | Path,
    episodes_per_env: int,
    horizon: int,
    split: str,
    seed: int = 0,
    include_unseen: bool = False,
) -> None:
    rng = np.random.default_rng(seed)
    env_ids = SEEN_ENVS + (UNSEEN_ENVS if include_unseen else ())

    graphs: list[np.ndarray] = []
    full_graphs: list[np.ndarray] = []
    envs: list[int] = []
    episodes: list[int] = []
    times: list[int] = []
    score_lefts: list[int] = []
    score_rights: list[int] = []
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    next_images: list[np.ndarray] = []
    next_masks: list[np.ndarray] = []

    for env_id in tqdm(env_ids, desc="Generating dataset"):
        for episode in range(episodes_per_env):
            state = initial_state(rng)
            for time_step in range(horizon):
                next_state, graph = step_state(state, env_id, rng)
                graphs.append(graph)
                full_graphs.append(graph_with_intervention(graph, env_id))
                envs.append(env_id)
                episodes.append(episode)
                times.append(time_step)
                score_lefts.append(int(state.score_left))
                score_rights.append(int(state.score_right))
                image, object_masks = render(state)
                next_image, next_object_masks = render(next_state)
                images.append(image)
                masks.append(object_masks)
                next_images.append(next_image)
                next_masks.append(next_object_masks)
                state = next_state

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "graph": np.asarray(graphs, dtype=np.uint8),
        "graph_with_env": np.asarray(full_graphs, dtype=np.uint8),
        "env": np.asarray(envs, dtype=np.int64),
        "episode": np.asarray(episodes, dtype=np.int64),
        "score_left": np.asarray(score_lefts, dtype=np.int64),
        "score_right": np.asarray(score_rights, dtype=np.int64),
        "t": np.asarray(times, dtype=np.int64),
        "split": np.asarray(split),
        "image": np.asarray(images, dtype=np.float32),
        "mask": np.asarray(masks, dtype=np.float32),
        "next_image": np.asarray(next_images, dtype=np.float32),
        "next_mask": np.asarray(next_masks, dtype=np.float32),
    }
    np.savez_compressed(path, **payload)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}
