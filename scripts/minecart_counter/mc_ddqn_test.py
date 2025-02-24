from typing import Any, Dict, cast

import gymnasium as gym
import minecart_counter
import numpy as np
from keras.api.models import Sequential
from keras.api.saving import load_model

from agents import DDQNAgent, TestLogger

MODEL_CLUSTERED_PATH = "models/MinecartCounter/clustered/glowing-resonance-28/1740061790_model____0.9119avg____0.9950max____0.0900min.keras"
MODEL_SCATTERED_PATH = "models/MinecartCounter/scattered/genial-brook-15/model_time_step_420321_episode_43750____0.9757avg____0.9800max____0.9650min.keras"

if __name__ == "__main__":

    env = gym.make(
        id="MinecartCounter-v2",
        render_mode="human",
        scatter_minecarts=True,
        render_raw_pixels=False,
        render_fps=3,
    )
    agent = DDQNAgent(env=env, obervation_normalization_type="image")

    model = load_model(filepath=MODEL_SCATTERED_PATH)
    model = cast(Sequential, model)

    log_vars = {
        "goal_reached_count": 0,
        "correct_goal_reached_count": 0,
        "truncated_count": 0,
    }

    def step_callback(
        observation: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[Any, Any],
        log_vars: Dict[str, Any],
    ):
        log_vars["goal_reached_count"] += 1 if info["goal_reached"] else 0
        log_vars["correct_goal_reached_count"] += 1 if info["correct_goal_reached"] else 0
        log_vars["truncated_count"] += 1 if info["truncated"] else 0

    test_logger = TestLogger(log_vars=log_vars, step_callback=step_callback)

    agent.test(model=model, episodes=20, test_logger=test_logger, epsilon=0.05)

    print(f"Goal reached count: {log_vars['goal_reached_count']}")
    print(f"Correct goal reached count: {log_vars['correct_goal_reached_count']}")
    print(f"Truncated count: {log_vars['truncated_count']}")
