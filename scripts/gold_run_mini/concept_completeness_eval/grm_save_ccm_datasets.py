import gold_run_mini
import gymnasium as gym
import numpy as np
from keras.api.models import Sequential
from keras.api.saving import load_model

from rl_ccm import create_base_and_eval_dataset

from .constants import DATASETS_CCM_DIR_PATH, ENV_LAVA_SPOTS, MODEL_OF_INTEREST_PATH


def grm_save_ccm_datasets():
    model: Sequential = load_model(MODEL_OF_INTEREST_PATH)  # type: ignore
    env = gym.make(
        id="GoldRunMini-v1",
        lava_spots=ENV_LAVA_SPOTS,
    )

    create_base_and_eval_dataset(
        size=10_000,
        env=env,
        model=model,
        normalization_callback="image",
        save_dir=DATASETS_CCM_DIR_PATH,
    )


if __name__ == "__main__":
    grm_save_ccm_datasets()
