from typing import Set, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import Env
from keras.api.models import Sequential
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from agents import ObservationNormalizationCallbacks


def get_base_and_eval_dataset(
    size: int, env: Env, model: Sequential, normalization_callback: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    terminated, truncated = False, False
    X = []
    Y = []
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
                y = int(np.argmax(q_values))

                X.append(observation)
                Y.append(y)
                seen_hashes.add(obs_hash)
                pbar.update(1)

                observation, _, terminated, truncated, _ = env.step(y)

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    return np.array(X_train), np.array(X_val), np.array(Y_train), np.array(Y_val)
