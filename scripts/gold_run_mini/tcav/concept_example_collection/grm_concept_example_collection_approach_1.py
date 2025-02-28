from datetime import datetime

import gold_run_mini
import gymnasium as gym

from rl_tcav import BinaryConceptExampleCollector

from .constants import ENV_LAVA_SPOTS, EXAMPLE_DATA_DIRECTORY_PATH, EXAMPLE_N
from .grm_concepts import get_grm_concepts

if __name__ == "__main__":
    # COLLECT EXAMPLES BY RANDOM POLICY PLAY

    env = gym.make(
        id="GoldRunMini-v1",
        lava_spots=ENV_LAVA_SPOTS,
    )

    batch_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    concept_list = get_grm_concepts()

    example_collector = BinaryConceptExampleCollector(
        concepts=concept_list,
        env=env,
        max_iter_per_concept=EXAMPLE_N,
        track_example_accumulation=True,
        normalization_callback="image",
    )

    example_collector.random_policy_play_collect_examples(example_n=EXAMPLE_N)
    example_collector.save_examples(
        directory_path=f"{EXAMPLE_DATA_DIRECTORY_PATH}/random_policy_play/{batch_tag}/",
    )
    example_collector.save_example_accumulation_data(
        directory_path=f"{EXAMPLE_DATA_DIRECTORY_PATH}/random_policy_play/{batch_tag}"
    )
