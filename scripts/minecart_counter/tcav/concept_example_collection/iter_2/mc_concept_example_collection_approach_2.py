import random
from datetime import datetime
from typing import cast

import gymnasium as gym
import minecart_counter
from keras.api.models import Sequential
from keras.api.saving import load_model

from rl_tcav import BinaryConceptExampleCollectorV2, ContinuousConceptExampleCollectorV2

from ...constants import MODEL_OF_INTEREST_PATH
from .constants import EXAMPLE_DATA_DIRECTORY_PATH, EXAMPLE_N, MAX_ITERATIONS
from .mc_concepts import get_mc_binary_concepts, get_mc_continuous_concepts

if __name__ == "__main__":
    # COLLECT EXAMPLES BY MODEL OF INTEREST GREEDY PLAY

    env = gym.make(
        id="MinecartCounter-v2",
        scatter_minecarts=True,
    )

    continuous_concepts = get_mc_continuous_concepts()
    binary_concepts = get_mc_binary_concepts()

    binary_concept_example_collector = BinaryConceptExampleCollectorV2(
        env=env,
        concepts=binary_concepts,
        max_iter=MAX_ITERATIONS,
        normalization_callback="image",
        track_example_accumulation=False,
    )

    continuous_concept_example_collector = ContinuousConceptExampleCollectorV2(
        env=env,
        concepts=continuous_concepts,
        max_iter=MAX_ITERATIONS,
        normalization_callback="image",
        track_example_accumulation=False,
    )

    model = load_model(MODEL_OF_INTEREST_PATH)
    model = cast(Sequential, model)

    binary_concept_example_collector.model_greedy_play_collect_examples(
        example_n=EXAMPLE_N, model=model
    )
    continuous_concept_example_collector.model_greedy_play_collect_examples(
        example_n=EXAMPLE_N, model=model
    )

    batch_tag = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{random.random()}"

    binary_concept_example_collector.save_examples(
        directory_path=f"{EXAMPLE_DATA_DIRECTORY_PATH}/model_of_interest_greedy_play/{batch_tag}/",
    )
    continuous_concept_example_collector.save_examples(
        directory_path=f"{EXAMPLE_DATA_DIRECTORY_PATH}/model_of_interest_greedy_play/{batch_tag}/",
    )
