import random
from datetime import datetime

import gem_collector
import gymnasium as gym

from rl_tcav import BinaryConceptExampleCollectorV2

from .constants import EXAMPLE_DATA_DIRECTORY_PATH, EXAMPLE_N
from .gc_concepts import get_gc_concepts


def appr1():
    env = gym.make(id="GemCollector-v3")

    concept_list = get_gc_concepts()

    example_collector = BinaryConceptExampleCollectorV2(
        concepts=concept_list,
        env=env,
        max_iter=100_000,
        track_example_accumulation=False,
        normalization_callback="image",
    )

    # COLLECT EXAMPLES BY RANDOM POLICY PLAY
    example_collector.random_policy_play_collect_examples(example_n=EXAMPLE_N)
    batch_tag = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{random.random()}"
    example_collector.save_examples(
        directory_path=f"{EXAMPLE_DATA_DIRECTORY_PATH}/random_policy_play/{batch_tag}/",
    )
