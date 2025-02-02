from typing import Tuple

import numpy as np
import scipy


class TrajectoryBuffer:

    def __init__(
        self,
        is_continuous_action_space: bool,
        observation_shape: Tuple[int, ...],
        num_actions: int,
        size: int,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> None:
        self.size = size
        if len(observation_shape) == 1:
            # Discrete/numerical observations (1D arrays)
            self.observation_buffer: np.ndarray = np.zeros(
                shape=(size, *observation_shape), dtype=np.float32
            )
        else:
            # Image observations (assumed to be multi-dimensional)
            self.observation_buffer: np.ndarray = np.zeros(
                shape=(size, *observation_shape), dtype=np.uint8
            )

        if is_continuous_action_space:
            self.action_buffer: np.ndarray = np.zeros((size, num_actions), dtype=np.float32)
        else:
            self.action_buffer: np.ndarray = np.zeros((size,), dtype=np.int32)

        self.advantage_buffer: np.ndarray = np.zeros(shape=size, dtype=np.float32)
        self.reward_buffer: np.ndarray = np.zeros(shape=size, dtype=np.float32)
        self.return_buffer: np.ndarray = np.zeros(shape=size, dtype=np.float32)
        self.value_buffer: np.ndarray = np.zeros(shape=size, dtype=np.float32)
        self.logprob_buffer: np.ndarray = np.zeros(shape=size, dtype=np.float32)
        self.gamma: float = gamma
        self.lam: float = lam
        self.pointer: int = 0
        self.trajectory_start_index: int = 0

    # Computes Generalized Advantage Estimation (GAE)
    def _discounted_cumulative_sums(self, sequence: np.ndarray, discount: float):
        if not (0.0 <= discount <= 1.0):
            raise ValueError('"discount" must be a float between 0 and 1.')

        return scipy.signal.lfilter([1], [1, float(-discount)], sequence[::-1], axis=0)[::-1]

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,  # Supports continuous actions
        reward: float,
        value: float,
        logprob: float,
    ) -> None:
        if self.pointer >= self.size:
            raise ValueError(
                "Buffer overflow. Consider increasing buffer size or handling buffer reset."
            )

        # Handle image vs. numerical observations properly
        if self.observation_buffer.dtype == np.uint8:  # Image case
            self.observation_buffer[self.pointer] = np.array(observation, dtype=np.uint8)
        else:  # Discrete/numerical observations
            self.observation_buffer[self.pointer] = np.array(observation, dtype=np.float32)

        # Handle discrete vs. continuous actions correctly
        if self.action_buffer.shape[1:] == ():  # Discrete case (scalar)
            self.action_buffer[self.pointer] = np.array(action, dtype=np.int32)
        else:  # Continuous case (vector)
            self.action_buffer[self.pointer] = np.array(action, dtype=np.float32)

        self.reward_buffer[self.pointer] = float(reward)
        self.value_buffer[self.pointer] = float(value)
        self.logprob_buffer[self.pointer] = float(logprob)

        self.pointer += 1

    def complete_episode_trajectory(self, last_value: float = 0) -> None:
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = self._discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = self._discounted_cumulative_sums(rewards, self.gamma)[:-1]

        self.trajectory_start_index = self.pointer

    def get_and_reset(self):
        if self.pointer != self.size:
            raise ValueError("get_and_empty called before buffer was full")

        advantage_mean = np.mean(self.advantage_buffer)
        advantage_std = np.std(self.advantage_buffer)
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std

        self.pointer, self.trajectory_start_index = 0, 0

        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprob_buffer,
        )
