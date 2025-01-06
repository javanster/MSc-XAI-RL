from typing import cast

import avocado_run
import gymnasium as gym
from keras.api.models import Sequential
from keras.api.saving import load_model

from agents import DoubleDQNAgent

if __name__ == "__main__":
    env = gym.make(id="AvocadoRun-v0", render_mode="human")

    agent = DoubleDQNAgent(env=env, obervation_normalization_type="image")

    model = load_model(
        filepath="models/AvocadoRun/flowing-music-74/1736170044_model____0.9744avg____1.0000max____0.8880min.keras"
    )
    model = cast(Sequential, model)

    agent.test(model=model, episodes=20)
