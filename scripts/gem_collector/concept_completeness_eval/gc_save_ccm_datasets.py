import gem_collector
import gymnasium as gym
import numpy as np
from keras.api.models import Sequential
from keras.api.saving import load_model

from rl_ccm import get_base_and_eval_dataset
from utils import ensure_directory_exists

from .constants import DATASETS_CCM_DIR_PATH, MODEL_OF_INTEREST_PATH


def gc_save_ccm_datasets():
    model: Sequential = load_model(MODEL_OF_INTEREST_PATH)  # type: ignore
    env = gym.make(id="GemCollector-v3")

    X_train, X_val, Y_train, Y_val = get_base_and_eval_dataset(
        size=10_000, env=env, model=model, normalization_callback="image"
    )

    ensure_directory_exists(DATASETS_CCM_DIR_PATH)
    np.save(f"{DATASETS_CCM_DIR_PATH}X_train_{len(X_train)}_examples.npy", X_train)
    np.save(f"{DATASETS_CCM_DIR_PATH}X_val_{len(X_val)}_examples.npy", X_val)
    np.save(f"{DATASETS_CCM_DIR_PATH}Y_train_{len(Y_train)}_labels.npy", Y_train)
    np.save(f"{DATASETS_CCM_DIR_PATH}Y_val_{len(Y_val)}_labels.npy", Y_val)

    X_train, X_val, Y_train, Y_val = get_base_and_eval_dataset(
        size=10_000, env=env, model=model, normalization_callback="image", all_q=True
    )

    ensure_directory_exists(DATASETS_CCM_DIR_PATH)
    np.save(f"{DATASETS_CCM_DIR_PATH}X_train_{len(X_train)}_all_q_examples.npy", X_train)
    np.save(f"{DATASETS_CCM_DIR_PATH}X_val_{len(X_val)}_all_q_examples.npy", X_val)
    np.save(f"{DATASETS_CCM_DIR_PATH}Y_train_{len(Y_train)}_all_q_labels.npy", Y_train)
    np.save(f"{DATASETS_CCM_DIR_PATH}Y_val_{len(Y_val)}_all_q_labels.npy", Y_val)


if __name__ == "__main__":
    gc_save_ccm_datasets()
