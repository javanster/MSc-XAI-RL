import random
from datetime import datetime
from typing import cast

import gold_run_mini
import gymnasium as gym
from keras.api.models import Sequential
from keras.api.saving import load_model

from rl_tcav import BinaryConceptExampleCollector

from ..constants import MORE_CAPABLE_MODEL_PATH
from .constants import ENV_LAVA_SPOTS, EXAMPLE_DATA_DIRECTORY_PATH, EXAMPLE_N
from .grm_concepts import get_grm_concepts

if __name__ == "__main__":
    # COLLECT EXAMPLES BY MORE CAPABLE MODEL EPSILON-GREEDY PLAY

    env = gym.make(
        id="GoldRunMini-v1",
        lava_spots=ENV_LAVA_SPOTS,
    )

    concept_list = get_grm_concepts()

    example_collector = BinaryConceptExampleCollector(
        concepts=concept_list,
        env=env,
        max_iter_per_concept=EXAMPLE_N,
        track_example_accumulation=True,
        normalization_callback="image",
    )

    # COLLECT EXAMPLE BY OPTIMAL MODEL PLAY WITH EPSILON = 0.05
    model = load_model(MORE_CAPABLE_MODEL_PATH)
    model = cast(Sequential, model)
    example_collector.model_epsilon_play_collect_examples(
        example_n=EXAMPLE_N, model=model, epsilon=0.05
    )
    batch_tag = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{random.random()}"

    example_collector.save_examples(
        directory_path=f"{EXAMPLE_DATA_DIRECTORY_PATH}/more_capable_model_epsilon0_005_play/{batch_tag}/",
    )
    example_collector.save_example_accumulation_data(
        directory_path=f"{EXAMPLE_DATA_DIRECTORY_PATH}/more_capable_model_epsilon0_005_play/{batch_tag}/"
    )
