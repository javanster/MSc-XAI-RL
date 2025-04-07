import random
from datetime import datetime
from typing import cast

import gem_collector
import gymnasium as gym
from keras.api.models import Sequential
from keras.api.saving import load_model

from rl_tcav import BinaryConceptExampleCollectorV2

from ...constants import MODEL_OF_INTEREST_PATH
from .constants import EXAMPLE_DATA_DIRECTORY_PATH, EXAMPLE_N
from .gc_concepts import get_gc_concepts


def appr2():
    env = gym.make(id="GemCollector-v3")

    concept_list = get_gc_concepts()
    example_collector = BinaryConceptExampleCollectorV2(
        concepts=concept_list,
        env=env,
        max_iter=100_000,
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
