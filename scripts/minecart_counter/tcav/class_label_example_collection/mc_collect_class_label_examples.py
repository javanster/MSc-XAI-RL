import random
from typing import cast

import gymnasium as gym
import minecart_counter
import numpy as np
from keras.api.models import Sequential
from keras.api.saving import load_model

from agents import ObservationNormalizationCallbacks
from rl_tcav import TcavClassLabelExampleCollector

from ..constants import MODEL_OF_INTEREST_PATH
from .constants import (
    CLASS_LABEL_EXAMPLE_SAVE_DIR_BASE_PATH,
    EXAMPLES_PER_OUTPUT_CLASS,
    MAX_ITERATIONS,
)

if __name__ == "__main__":
    env = gym.make(
        id="MinecartCounter-v2",
        scatter_minecarts=True,
    )

    def random_action_callback(_: np.ndarray, __: int) -> int:
        return random.randint(0, env.action_space.n - 1)

    random_class_label_collector = TcavClassLabelExampleCollector(
        env=env,
        action_callback=random_action_callback,
    )

    random_class_label_collector.collect_examples(
        examples_per_output_class=EXAMPLES_PER_OUTPUT_CLASS, max_iterations=MAX_ITERATIONS
    )

    random_class_label_collector.save_examples(
        f"{CLASS_LABEL_EXAMPLE_SAVE_DIR_BASE_PATH}random_policy/"
    )

    moi: Sequential = load_model(MODEL_OF_INTEREST_PATH)  # type: ignore

    def moi_action_callback(observation: np.ndarray, _: int) -> int:
        observation_reshaped = ObservationNormalizationCallbacks.normalize_images(
            np.array(observation).reshape(-1, *observation.shape)
        )
        q_values = moi.predict(observation_reshaped)[0]
        action = int(np.argmax(q_values))
        return action

    moi_class_label_collector = TcavClassLabelExampleCollector(
        env=env,
        action_callback=moi_action_callback,
    )

    moi_class_label_collector.collect_examples(
        examples_per_output_class=EXAMPLES_PER_OUTPUT_CLASS, max_iterations=MAX_ITERATIONS
    )

    moi_class_label_collector.save_examples(
        f"{CLASS_LABEL_EXAMPLE_SAVE_DIR_BASE_PATH}model_of_interest/"
    )
