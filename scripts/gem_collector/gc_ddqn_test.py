import argparse
from typing import Any, Dict, cast

import gem_collector
import gymnasium as gym
import numpy as np
from keras.api.models import Sequential
from keras.api.saving import load_model

from agents import DDQNAgent, TestLogger

OPTIMAL_POLICY_MODEL_FILE_PATH = "models/GemCollector/iconic-sweep-77/1738860819_model____0.6935avg____0.7910max____0.4690min.keras"
SUB_OPTIMAL_POLICY_MODEL_FILE_PATH = "models/GemCollector/denim-sweep-56/1738828803_model____0.3766avg____0.6576max____0.1570min.keras"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run GemCollector with different policy models.")
    parser.add_argument(
        "--model",
        choices=["optimal", "sub-optimal"],
        default="sub-optimal",
        help="Choose which model to run: 'optimal' or 'sub-optimal' (default: sub-optimal).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    log_vars = {
        "aquamarines_collected_count": 0,
        "amethysts_collected_count": 0,
        "emeralds_collected_count": 0,
        "rocks_collected_count": 0,
        "lava_termination_count": 0,
        "truncated_count": 0,
        "total_reward": 0,
    }

    def step_callback(
        observation: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[Any, Any],
        log_vars: Dict[str, Any],
    ):
        log_vars["aquamarines_collected_count"] += info["aquamarine_collected"]
        log_vars["amethysts_collected_count"] += info["amethyst_collected"]
        log_vars["emeralds_collected_count"] += info["emerald_collected"]
        log_vars["rocks_collected_count"] += info["rocks_collected"]
        log_vars["lava_termination_count"] += 1 if info["lava_collision"] else 0
        log_vars["truncated_count"] += 1 if truncated else 0
        log_vars["total_reward"] += info["reward"]

    print(f"Testing with {args.model} model")

    env = gym.make(id="GemCollector-v3", render_mode="human", show_raw_pixels=False)
    agent = DDQNAgent(env=env, obervation_normalization_type="image")

    model = load_model(filepath=OPTIMAL_POLICY_MODEL_FILE_PATH)
    model = cast(Sequential, model)

    test_logger = TestLogger(log_vars=log_vars, step_callback=step_callback)

    EPISODES = 1

    agent.test(model=model, episodes=EPISODES, test_logger=test_logger, epsilon=-1)

    print(log_vars)

    print(f"Avg reward per episode: {log_vars['total_reward'] / EPISODES}")
