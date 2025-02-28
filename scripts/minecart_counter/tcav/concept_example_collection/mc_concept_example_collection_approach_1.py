import random
from datetime import datetime

import gem_collector
import gymnasium as gym

from rl_tcav import BinaryConceptExampleCollector, ContinuousConceptExampleCollector

from .constants import EXAMPLE_DATA_DIRECTORY_PATH, EXAMPLE_N
from .mc_concepts import get_mc_binary_concepts, get_mc_continuous_concepts

if __name__ == "__main__":
    # COLLECT EXAMPLES BY RANDOM POLICY PLAY

    env = gym.make(
        id="MinecartCounter-v2",
        scatter_minecarts=True,
    )

    continuous_concepts = get_mc_continuous_concepts()
    binary_concepts = get_mc_binary_concepts()

    binary_concept_example_collector = BinaryConceptExampleCollector(
        env=env,
        concepts=binary_concepts,
        max_iter_per_concept=EXAMPLE_N,
        normalization_callback="image",
        track_example_accumulation=True,
    )

    continuous_concept_example_collector = ContinuousConceptExampleCollector(
        env=env,
        concepts=continuous_concepts,
        max_iter_per_concept=EXAMPLE_N,
        normalization_callback="image",
        track_example_accumulation=True,
    )

    binary_concept_example_collector.random_policy_play_collect_examples(example_n=EXAMPLE_N)
    continuous_concept_example_collector.random_policy_play_collect_examples(example_n=EXAMPLE_N)

    batch_tag = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{random.random()}"

    binary_concept_example_collector.save_examples(
        directory_path=f"{EXAMPLE_DATA_DIRECTORY_PATH}/random_policy_play/{batch_tag}/",
    )
    binary_concept_example_collector.save_example_accumulation_data(
        directory_path=f"{EXAMPLE_DATA_DIRECTORY_PATH}/random_policy_play/{batch_tag}"
    )
    continuous_concept_example_collector.save_examples(
        directory_path=f"{EXAMPLE_DATA_DIRECTORY_PATH}/random_policy_play/{batch_tag}/",
    )
    continuous_concept_example_collector.save_example_accumulation_data(
        directory_path=f"{EXAMPLE_DATA_DIRECTORY_PATH}/random_policy_play/{batch_tag}"
    )
