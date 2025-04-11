import gem_collector
import gymnasium as gym
import numpy as np
from keras.api.models import Sequential
from keras.api.saving import load_model

from rl_ccm import create_base_and_eval_dataset

from .constants import DATASETS_CCM_DIR_PATH, MODEL_OF_INTEREST_PATH


def gc_save_ccm_datasets():
    model: Sequential = load_model(MODEL_OF_INTEREST_PATH)  # type: ignore
    env = gym.make(id="GemCollector-v3")

    create_base_and_eval_dataset(
        size=10_000,
        env=env,
        model=model,
        normalization_callback="image",
        save_dir=DATASETS_CCM_DIR_PATH,
    )


if __name__ == "__main__":
    gc_save_ccm_datasets()
