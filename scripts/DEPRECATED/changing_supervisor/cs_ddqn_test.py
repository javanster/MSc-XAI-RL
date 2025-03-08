from typing import cast

import changing_supervisor
import gymnasium as gym
from keras.api.models import Sequential
from keras.api.saving import load_model

from agents import DDQNAgent

MODEL_FILE_PATH = "models/ChangingSupervisor/likely-pine-28/1740034304_model___-0.2385avg___-0.0150max___-0.5125min.keras"


if __name__ == "__main__":

    env = gym.make(id="ChangingSupervisor-v1", render_mode="human")
    agent = DDQNAgent(env=env, obervation_normalization_type="image")

    model = load_model(filepath=MODEL_FILE_PATH)
    model = cast(Sequential, model)

    agent.test(model=model, episodes=10)
