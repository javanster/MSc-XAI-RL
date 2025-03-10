import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from rl_tcav import BCEValidationSetCurator

random.seed(28)
np.random.seed(28)
tf.random.set_seed(28)
from .constants import (
    CONCEPT_DATA_APPROACH_PATH_BASE,
    CONCEPT_NAMES,
    CONCEPT_PREFIXES,
    VALIDATION_SET_TARGET_SIZES,
)

if __name__ == "__main__":

    for concept_i, concept_prefix in enumerate(CONCEPT_PREFIXES):
        bce_validation_set_curator = BCEValidationSetCurator(
            concept_data_approach_path_base=CONCEPT_DATA_APPROACH_PATH_BASE,
            concept_file_prefix=concept_prefix,
            concept_name=CONCEPT_NAMES[concept_i],
            environment_name="gold_run_mini",
            target_size=VALIDATION_SET_TARGET_SIZES[concept_i],
        )

        bce_validation_set_curator.curate_validation_set()
