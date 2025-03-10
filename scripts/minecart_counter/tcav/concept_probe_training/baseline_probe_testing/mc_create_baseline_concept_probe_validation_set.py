import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from rl_tcav import BCEValidationSetCurator, CCEValidationSetCurator

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

    cce_validation_set_curator = CCEValidationSetCurator(
        concept_data_approach_path_base=CONCEPT_DATA_APPROACH_PATH_BASE,
        concept_file_prefix=CONCEPT_PREFIXES[0],
        concept_name=CONCEPT_NAMES[0],
        environment_name="minecart_counter",
        target_size=VALIDATION_SET_TARGET_SIZES[0],
    )

    bce_validation_set_curator = BCEValidationSetCurator(
        concept_data_approach_path_base=CONCEPT_DATA_APPROACH_PATH_BASE,
        concept_file_prefix=CONCEPT_PREFIXES[1],
        concept_name=CONCEPT_NAMES[1],
        environment_name="minecart_counter",
        target_size=VALIDATION_SET_TARGET_SIZES[1],
    )

    cce_validation_set_curator.curate_validation_set()
    bce_validation_set_curator.curate_validation_set()
