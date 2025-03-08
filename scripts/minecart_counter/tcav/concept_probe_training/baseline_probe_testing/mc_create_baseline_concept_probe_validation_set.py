import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from rl_tcav import BCEValidationSetCurator, CCEValidationSetCurator

random.seed(28)
np.random.seed(28)
tf.random.set_seed(28)


if __name__ == "__main__":

    concept_data_approach_path_base = "rl_tcav_data/concept_examples/minecart_counter/"
    target_size = 10_000

    bce_validation_set_curator = BCEValidationSetCurator(
        concept_data_approach_path_base=concept_data_approach_path_base,
        concept_file_prefix="binary_concept_minecart_1_left",
        concept_name="minecart_1_left",
        environment_name="minecart_counter",
        target_size=target_size,
    )
    bce_validation_set_curator.curate_validation_set()

    cce_validation_set_curator = CCEValidationSetCurator(
        concept_data_approach_path_base=concept_data_approach_path_base,
        concept_file_prefix="continuous_concept_minecarts_n",
        concept_name="minecarts_n",
        environment_name="minecart_counter",
        target_size=target_size,
    )
    cce_validation_set_curator.curate_validation_set()
