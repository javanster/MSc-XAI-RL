import argparse
from typing import cast

import box_escape
import gymnasium as gym
from keras.api.models import Sequential
from keras.api.saving import load_model
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper

from agents import DDQNAgent

if __name__ == "__main__":

    model_path = "models/BoxEscape/gentle-dew-101/1739424389_model____0.0832avg____0.9960max___-0.8160min.keras"

    env = gym.make(
        id="BoxEscape-v1",
        render_mode="human",
        fully_observable=True,
        curriculum_level=1,
    )
    env = ImgObsWrapper(RGBImgObsWrapper(env))
    agent = DDQNAgent(env=env, obervation_normalization_type="image")

    model = load_model(filepath=model_path)
    model = cast(Sequential, model)

    agent.test(model=model, episodes=1)
