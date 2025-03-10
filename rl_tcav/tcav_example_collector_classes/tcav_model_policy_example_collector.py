from typing import Callable, Dict, List, cast

import numpy as np
from gymnasium import Env
from keras.api.models import Sequential
from tqdm import tqdm

from .tcav_example_collector import TcavExampleCollector


class TcavModelPolicyExampleCollector(TcavExampleCollector):

    def __init__(
        self,
        env: Env,
        model: Sequential,
        observation_normalization_callback: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        super().__init__(output_classes_n=env.action_space.n)
        self.env: Env = env
        self.model: Sequential = model
        self.observation_normalization_callback: Callable[[np.ndarray], np.ndarray] = (
            observation_normalization_callback
        )

    def _is_done(self, examples_per_output_class: int):
        return all(
            len(examples) == examples_per_output_class
            for examples in self.collected_examples.values()
        )

    def collect_examples(self, examples_per_output_class: int) -> None:

        with tqdm(
            total=examples_per_output_class * self.env.action_space.n, unit="example"
        ) as pbar:
            while not self._is_done(examples_per_output_class=examples_per_output_class):
                observation, _ = self.env.reset()
                observation = cast(np.ndarray, observation)
                terminated: bool = False
                truncated: bool = False

                while not terminated and not truncated:

                    observation_reshaped = self.observation_normalization_callback(
                        np.array(observation).reshape(-1, *observation.shape)
                    )
                    q_values = self.model.predict(observation_reshaped)[0]
                    action = int(np.argmax(q_values))

                    examples = self.collected_examples[action]
                    if len(examples) < examples_per_output_class:
                        examples.append(observation)
                        pbar.update(1)

                        if self._is_done(examples_per_output_class=examples_per_output_class):
                            return

                    observation, _, terminated, truncated, _ = self.env.step(action=action)
                    observation = cast(np.ndarray, observation)
                    terminated = cast(bool, terminated)
                    truncated = cast(bool, truncated)

    def save_examples(self, directory_path: str) -> None:
        for output_class in self.collected_examples.keys():
            self._save_examples_for_output_class(
                directory_path=directory_path, output_class=output_class
            )

    def _save_examples_for_output_class(self, directory_path: str, output_class: int) -> None:
        collected_examples = self.collected_examples[output_class]

        if len(collected_examples) == 0:
            print(f"No examples of output class {output_class} to save, returning...")
            return

        self._ensure_save_directory_exists(directory_path=directory_path)
        file_path = (
            f"{directory_path}/{len(collected_examples)}_examples_output_class_{output_class}.npy"
        )
        array = np.array(collected_examples)
        np.save(file_path, array)
        print(f"Examples for output class {output_class} successfully saved to {file_path}.")
