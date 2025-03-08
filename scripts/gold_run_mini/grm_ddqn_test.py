from typing import Any, Dict, cast

import gold_run_mini
import gymnasium as gym
import numpy as np
from keras.api.models import Sequential
from keras.api.saving import load_model

from agents import DDQNAgent, TestLogger

MODEL_OF_INTEREST_PATH = "models/GoldRunMini/sub-competent/firm-mountain-13/model_time_step_457199_episode_15800____0.4878avg____0.5000max____0.4532min.keras"
MORE_CAPABLE_MODEL_PATH = "models/GoldRunMini/competent/firm-salad-11/model_time_step_390195_episode_3100____0.8544avg____0.9910max___-0.2000min.keras"


if __name__ == "__main__":

    log_vars = {
        "lava_stepped_on_count": 0,
        "gold_picked_up_count": 0,
        "went_to_next_room_count": 0,
        "exited_in_final_passage_count": 0,
        "exited_in_early_termination_passage_count": 0,
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
        log_vars["lava_stepped_on_count"] += info["lava_stepped_on"]
        log_vars["gold_picked_up_count"] += info["gold_picked_up"]
        log_vars["went_to_next_room_count"] += 1 if info["went_to_next_room"] else 0
        log_vars["exited_in_final_passage_count"] += 1 if info["exited_in_final_passage"] else 0
        log_vars["exited_in_early_termination_passage_count"] += (
            1 if info["exited_in_early_termination_passage"] else 0
        )
        log_vars["truncated_count"] += 1 if truncated else 0

    env = gym.make(
        id="GoldRunMini-v1",
        render_mode="human",
        render_raw_pixels=False,
        disable_early_termination=False,
        no_lava_termination=False,
        only_second_room=False,
        lava_spots=8,
    )
    input_shape = env.observation_space.shape
    output_shape = env.action_space.n
    agent = DDQNAgent(env=env, obervation_normalization_type="image")
    model = load_model(MODEL_OF_INTEREST_PATH)
    model = cast(Sequential, model)
    test_logger = TestLogger(log_vars=log_vars, step_callback=step_callback)

    agent.test(env=env, episodes=20, model=model, test_logger=test_logger, epsilon=0.05)

    print(log_vars)
