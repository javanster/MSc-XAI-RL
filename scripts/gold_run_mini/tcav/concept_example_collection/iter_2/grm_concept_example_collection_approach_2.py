import random
from datetime import datetime
from typing import cast

import gold_run_mini
import gymnasium as gym
from keras.api.models import Sequential
from keras.api.saving import load_model

from rl_tcav import BinaryConceptExampleCollectorV2

from ...constants import MODEL_OF_INTEREST_PATH
from .constants import ENV_LAVA_SPOTS, EXAMPLE_DATA_DIRECTORY_PATH, EXAMPLE_N, MAX_ITERATIONS
from .grm_concepts import get_grm_concepts


def appr2():
    env = gym.make(
        id="GoldRunMini-v1",
        lava_spots=ENV_LAVA_SPOTS,
    )

    concept_list = get_grm_concepts()
    example_collector = BinaryConceptExampleCollectorV2(
        concepts=concept_list,
        env=env,
        max_iter=MAX_ITERATIONS,
        track_example_accumulation=False,
        normalization_callback="image",
    )

    # COLLECT EXAMPLE BY MODEL OF INTEREST PLAY
    model = load_model(MODEL_OF_INTEREST_PATH)
    model = cast(Sequential, model)
    example_collector.model_greedy_play_collect_examples(example_n=EXAMPLE_N, model=model)
    batch_tag = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{random.random()}"
    example_collector.save_examples(
        directory_path=f"{EXAMPLE_DATA_DIRECTORY_PATH}/model_of_interest_greedy_play/{batch_tag}/",
    )


if __name__ == "__main__":
    appr2()
