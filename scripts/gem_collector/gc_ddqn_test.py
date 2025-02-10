import argparse
from typing import cast

import gem_collector
import gymnasium as gym
from keras.api.models import Sequential
from keras.api.saving import load_model

from agents import DDQNAgent

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

    model_path = (
        OPTIMAL_POLICY_MODEL_FILE_PATH
        if args.model == "optimal"
        else SUB_OPTIMAL_POLICY_MODEL_FILE_PATH
    )

    print(f"Testing with {args.model} model")

    env = gym.make(id="GemCollector-v3", render_mode="human", show_raw_pixels=False)
    agent = DDQNAgent(env=env, obervation_normalization_type="image")

    model = load_model(filepath=model_path)
    model = cast(Sequential, model)

    agent.test(model=model, episodes=1)
