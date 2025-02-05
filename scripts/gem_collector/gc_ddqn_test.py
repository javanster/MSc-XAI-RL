from typing import cast

import gem_collector
import gymnasium as gym
from keras.api.models import Sequential
from keras.api.saving import load_model

from agents import DDQNAgent

if __name__ == "__main__":
    env = gym.make(id="GemCollector-v3", render_mode="human", show_raw_pixels=False)
    agent = DDQNAgent(env=env, obervation_normalization_type="image")
    model = load_model(
        filepath="models/GemCollector/northern-sweep-19/1738791493_model____0.3843avg____0.5330max____0.2126min.keras"
    )
    model = cast(Sequential, model)
    agent.test(model=model, episodes=1)
