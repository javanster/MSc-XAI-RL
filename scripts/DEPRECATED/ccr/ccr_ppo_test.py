from typing import cast

import ccr
import gymnasium as gym
import numpy as np
import tensorflow as tf
from ccr.wrappers import FrameStack
from keras.api.models import Model
from keras.api.saving import load_model

import wandb
from agents import PPOAgent

from .ccr_actor_critic_model import scale_inputs

if __name__ == "__main__":

    env = gym.make("CCR-v5", render_mode="human")
    env = FrameStack(env=env, k=4)

    agent = PPOAgent(
        env=env,
    )
    model = load_model(
        custom_objects={"scale_inputs": scale_inputs},
        filepath="ppo_models/CCR/generous-terrain-18/1739837537_model__332.2872avg__454.6982max__205.6938min.keras",
    )
    model = cast(Model, model)
    agent.test(env=env, model=model)
