import random
from typing import Set

import gymnasium as gym
import numpy as np
import tensorflow as tf
from gymnasium import Env
from keras.api.models import Sequential
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from agents import ObservationNormalizationCallbacks
from utils import ensure_directory_exists


def create_base_and_eval_dataset(
    size: int,
    env: Env,
    model: Sequential,
    normalization_callback: str,
    save_dir: str,
) -> None:
    """
    Generate and save a dataset of observations, actions, and Q-values using a trained DQN Sequential Keras model.

    This function creates a dataset by interacting with an environment using the model's policy,
    collecting unique observations along with:
      - the action selected by taking argmax(Q),
      - the full Q-values output by the model.

    The collected dataset is then split into training and validation sets and saved to disk as .npy files.

    Parameters
    ----------
    size : int
        Number of unique observations to collect.
    env : Env
        A Gymnasium-compatible environment instance.
    model : Sequential
        A trained Keras model that predicts Q-values given an observation.
    normalization_callback : str
        The name of the normalization function to apply to observations before passing to the model.
        Must be a key in ObservationNormalizationCallbacks.normalization_callbacks.
    save_dir : str
        Directory where the resulting .npy files will be saved. Will be created if it does not exist.
    """

    random.seed(28)
    np.random.seed(28)
    tf.random.set_seed(28)

    terminated, truncated = False, False
    X = []
    Y_max_q = []
    Y_all_q = []
    seen_hashes: Set[int] = set()

    if (
        normalization_callback
        not in ObservationNormalizationCallbacks.normalization_callbacks.keys()
    ):
        raise ValueError(
            f"Provided normalization_callback is not valid, please provide one of the following: "
            f"{[callback_name for callback_name in ObservationNormalizationCallbacks.normalization_callbacks.keys()]}"
        )
    normalization_callback_func = ObservationNormalizationCallbacks.normalization_callbacks[
        normalization_callback
    ]

    observation, _ = env.reset()

    with tqdm(total=size, unit="observation") as pbar:
        while len(X) < size:
            if terminated or truncated:
                observation, _ = env.reset()
                terminated, truncated = False, False

            else:
                obs_hash = hash(observation.tobytes())
                if obs_hash in seen_hashes:
                    observation, _, terminated, truncated, _ = env.step(env.action_space.sample())
                    continue

                observation_reshaped = normalization_callback_func(
                    np.array(observation).reshape(-1, *observation.shape)
                )
                q_values = model.predict(observation_reshaped)[0]
                action = int(np.argmax(q_values))

                X.append(observation)
                Y_max_q.append(action)
                Y_all_q.append(q_values)
                seen_hashes.add(obs_hash)
                pbar.update(1)

                observation, _, terminated, truncated, _ = env.step(action)

    X_train, X_val, Y_max_q_train, Y_max_q_val, Y_all_q_train, Y_all_q_val = train_test_split(
        X, Y_max_q, Y_all_q, test_size=0.2, random_state=42, stratify=Y_max_q
    )

    ensure_directory_exists(save_dir)
    np.save(f"{save_dir}X_train_{len(X_train)}_examples.npy", X_train)
    np.save(f"{save_dir}X_val_{len(X_val)}_examples.npy", X_val)
    np.save(f"{save_dir}Y_max_q_train_{len(Y_max_q_train)}_examples.npy", Y_max_q_train)
    np.save(f"{save_dir}Y_max_q_val_{len(Y_max_q_val)}_examples.npy", Y_max_q_val)
    np.save(f"{save_dir}Y_all_q_train_{len(Y_all_q_train)}_examples.npy", Y_all_q_train)
    np.save(f"{save_dir}Y_all_q_val_{len(Y_all_q_val)}_examples.npy", Y_all_q_val)
