from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spartan_pong.config import N_OBJECTS, SEEN_ENVS, UNSEEN_ENVS

# Environment variables
FIELD_W = 1.0
FIELD_H = 1.0
LEFT_X = 0.06
RIGHT_X = 0.94
PADDLE_H = 0.22
PADDLE_W = 0.035
BALL_R = 0.035


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


def _clip_y(y: float) -> float:
    # Clip paddle center y to stay within the field. We allow the ball to go out of bounds.
    return float(np.clip(y, PADDLE_H / 2.0, FIELD_H - PADDLE_H / 2.0))


def initial_state(rng: np.random.Generator) -> PongState:
    # Define the inital state of the game. The ball starts in the middle with a random velocity, and the paddles start at random heights.
    vx = rng.choice([-1.0, 1.0]) * rng.uniform(0.018, 0.026)
    vy = rng.uniform(-0.018, 0.018)
    return PongState(
        left_y=float(rng.uniform(0.25, 0.75)),
        right_y=float(rng.uniform(0.25, 0.75)),
        ball_x=float(rng.uniform(0.35, 0.65)),
        ball_y=float(rng.uniform(0.25, 0.75)),
        ball_vx=float(vx),
        ball_vy=float(vy),
    )


def step_state(state: PongState, env_id: int) -> tuple[PongState, np.ndarray]:
    """
    Do one transition and return the next state plus the local causal graph.
    Graph convention is row=child at t+1, column=parent at t. This matches the
    paper's A_ij meaning token j is a parent of prediction i.
    """
    # The graph is defined based on the known dynamics of the game and the env_id. It should be deterministic and not depend on random seeds.
    # 0: left paddle, 1: right paddle, 2: ball, 3: score
    graph = np.zeros((N_OBJECTS, N_OBJECTS), dtype=np.uint8)
    graph[0, 2] = 1  # ball → left paddle (always)
    graph[1, 2] = 1  # ball → right paddle (always)

    # The paddles move towards the ball with a fixed gain.
    # Some environments have different movement logic
    left_gain = -0.045 if env_id in (2, 9) else 0.045
    right_gain = -0.045 if env_id == 10 else 0.045
    target_right_y = 1.0 - state.ball_y if env_id == 3 else state.ball_y
    left_y = _clip_y(state.left_y + left_gain * np.sign(state.ball_y - state.left_y))
    right_y = _clip_y(state.right_y + right_gain * np.sign(target_right_y - state.right_y))

    # Apply environment-specific ball physics
    vx = state.ball_vx
    vy = state.ball_vy
    x = state.ball_x
    y = state.ball_y

    if env_id in (1, 9):
        # Ball follows a sine wave pattern in the x direction.
        vy += 0.004 * np.sin(8.0 * np.pi * x)
    if env_id in (6, 7, 10) and x < 0.5:
        # Gravity on the left half of the field.
        # 0 is at the top of the field, so we add to vy to pull the ball down.
        vy += 0.0025
    if env_id in (4, 8) and 0.38 <= x <= 0.62:
        # strong friction in the middle of the field
        vx *= 0.93
        vy *= 0.93

    # Update ball position based on velocity, and check for collisions with walls and paddles
    x += vx
    y += vy

    # Collision with top and bottom walls
    if y <= BALL_R or y >= FIELD_H - BALL_R:
        y = float(np.clip(y, BALL_R, FIELD_H - BALL_R))
        vy *= -1.0

    # Collision with paddles
    hit_left = vx < 0 and x <= LEFT_X + PADDLE_W and abs(y - left_y) <= PADDLE_H / 2.0
    hit_right = vx > 0 and x >= RIGHT_X - PADDLE_W and abs(y - right_y) <= PADDLE_H / 2.0
    # If the ball hits a paddle, reverse the x velocity, add spin based on where it hits the paddle.
    # add causal edge
    # push the ball outside the paddle to prevent sticking
    # env 5, 7 has stronger bounce, and env 8 has stronger bounce on the left paddle only
    if hit_left or hit_right:
        graph[2, 0 if hit_left else 1] = 1  # ball → left/right paddle (conditional!)
        x = LEFT_X + PADDLE_W + BALL_R if hit_left else RIGHT_X - PADDLE_W - BALL_R
        bounce_scale = 1.35 if env_id in (5, 7) or (env_id == 8 and hit_left) else 1.0
        vx = -vx * bounce_scale
        paddle_y = left_y if hit_left else right_y
        vy += 0.04 * (y - paddle_y)

    # Check for scoring, reset, add causal edge, update scores
    score_left = state.score_left
    score_right = state.score_right
    if x < -BALL_R or x > FIELD_W + BALL_R:
        graph[3, 2] = 1  # score → ball (scoring depends on ball going out of bounds)
        if x < -BALL_R:
            score_right += 1.0
        else:
            score_left += 1.0
        reset = initial_state(
            np.random.default_rng(int((score_left + 3) * 997 + (score_right + 5) * 991))
        )
        x, y, vx, vy = reset.ball_x, reset.ball_y, reset.ball_vx, reset.ball_vy

    next_state = PongState(
        left_y, right_y, float(x), float(y), float(vx), float(vy), score_left, score_right
    )
    return next_state, graph


def intervention_edges(env_id: int) -> np.ndarray:
    """
    Return object children whose dynamics are directly changed by the env token.
    This is used to define the full graph with the env token as an additional parent.
    So it returns a [N_OBJECTS,N_OBJECTS+1] vector where 1 means the env token (the last column) is a parent of the object (the row).
    [4,5] in this case.
    """
    edges = np.zeros(N_OBJECTS, dtype=np.uint8)
    if env_id in (2, 9):
        edges[0] = 1  # env → left_paddle
    if env_id in (3, 10):
        edges[1] = 1  # env → right_paddle
    if env_id in (1, 4, 5, 6, 7, 8, 9, 10):
        edges[2] = 1  # env → ball
    return edges


def graph_with_intervention(graph: np.ndarray, env_id: int) -> np.ndarray:
    # Combine the local causal graph with the intervention edges to get the full graph with the env token as an additional parent.
    full = np.zeros((N_OBJECTS, N_OBJECTS + 1), dtype=np.uint8)
    full[:, :N_OBJECTS] = graph
    full[:, N_OBJECTS] = intervention_edges(env_id)
    return full


def render(
    state: PongState, size: int = 32, include_ball_trace: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Render the current state as a 32x32 RGB image and object masks.
    The object masks are binary masks for each object (left paddle, right paddle, ball, score) with shape [N_OBJECTS, size, size].
    """
    # We render the paddles as rectangles, the ball as a circle, and the score as a bar at the top. The background is black.
    image = np.zeros((size, size, 3), dtype=np.float32)
    masks = np.zeros((N_OBJECTS, size, size), dtype=np.float32)

    # Helper function to draw rectangles for paddles and ball. It also updates the corresponding object mask.
    def rect(
        obj: int, cx: float, cy: float, w: float, h: float, color: tuple[float, float, float]
    ) -> None:
        # Convert from center coordinates and width/height to pixel coordinates, and clip to image boundaries
        x0 = max(0, int((cx - w / 2) * size))
        x1 = min(size, int((cx + w / 2) * size) + 1)
        y0 = max(0, int((cy - h / 2) * size))
        y1 = min(size, int((cy + h / 2) * size) + 1)
        image[y0:y1, x0:x1] = color
        # Update the object mask for this object. We set it to 1.0 in the region of the rectangle.
        masks[obj, y0:y1, x0:x1] = 1.0

    # Draw left paddle, right paddle, ball, and score. The score is represented as a bar at the top of the image, where the length of the bar corresponds to the total score.
    rect(0, LEFT_X, state.left_y, PADDLE_W, PADDLE_H, (0.2, 0.6, 1.0))
    rect(1, RIGHT_X, state.right_y, PADDLE_W, PADDLE_H, (1.0, 0.5, 0.2))
    # This is used to help the model capture the velocity of the ball,
    # since the ball is small and can be hard to track. We render a faint circle at the previous position of the ball.
    if include_ball_trace:
        prev_x = float(np.clip(state.ball_x - state.ball_vx, BALL_R, FIELD_W - BALL_R))
        prev_y = float(np.clip(state.ball_y - state.ball_vy, BALL_R, FIELD_H - BALL_R))
        rect(2, prev_x, prev_y, BALL_R * 2, BALL_R * 2, (0.25, 0.75, 1.0))
    # Draw the ball
    rect(2, state.ball_x, state.ball_y, BALL_R * 2, BALL_R * 2, (1.0, 1.0, 1.0))
    # Score bar
    image[0:2, :, 1] = min(1.0, (state.score_left + state.score_right) / 10.0)
    # masks for score object is the same as the score bar in the image
    masks[3, 0:2, :] = 1.0
    return image, masks


def generate_dataset(
    out: str | Path,
    episodes_per_env: int,  # number of games per env
    horizon: int,  # number of transitions per game
    split: str,
    seed: int = 0,
    include_unseen: bool = False,
) -> None:
    # Generate a dataset of transitions by simulating the Pong environment with defined dynamics.
    # Seed
    rng = np.random.default_rng(seed)
    # Generate data for specified environments
    env_ids = SEEN_ENVS + (UNSEEN_ENVS if include_unseen else ())

    # Define lists to store the dataset. We will convert them to numpy arrays at the end and save as .npz.
    graphs: list[np.ndarray] = []
    full_graphs: list[np.ndarray] = []
    envs: list[int] = []
    episodes: list[int] = []
    times: list[int] = []
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    next_images: list[np.ndarray] = []
    next_masks: list[np.ndarray] = []

    # Simulate the environment and collect data
    for env_id in env_ids:
        # N episodes per env
        for episode in range(episodes_per_env):
            state = initial_state(rng)
            for time_step in range(horizon):
                next_state, graph = step_state(state, env_id)
                # Store the transition data.
                graphs.append(graph)  # object graph
                full_graphs.append(graph_with_intervention(graph, env_id))  # object + env graph
                envs.append(env_id)  # which env
                episodes.append(episode)  # which game
                times.append(time_step)  # which timestep
                image, object_masks = render(
                    state
                )  # render the current state as an image and object masks
                next_image, next_object_masks = render(
                    next_state
                )  # render the next state as an image and object masks
                # store the images
                images.append(image)
                masks.append(object_masks)
                next_images.append(next_image)
                next_masks.append(next_object_masks)
                state = next_state

    # Save the dataset as a compressed .npz file.
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "graph": np.asarray(graphs, dtype=np.uint8),
        "graph_with_env": np.asarray(full_graphs, dtype=np.uint8),
        "env": np.asarray(envs, dtype=np.int64),
        "episode": np.asarray(episodes, dtype=np.int64),
        "t": np.asarray(times, dtype=np.int64),
        "split": np.asarray(split),
        "image": np.asarray(images, dtype=np.float32),
        "mask": np.asarray(masks, dtype=np.float32),
        "next_image": np.asarray(next_images, dtype=np.float32),
        "next_mask": np.asarray(next_masks, dtype=np.float32),
    }
    np.savez_compressed(path, **payload)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    # load the dataset
    data = np.load(path)
    return {key: data[key] for key in data.files}
