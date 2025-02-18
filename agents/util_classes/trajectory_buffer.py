from typing import Tuple

import numpy as np
import scipy
import tensorflow as tf


class TrajectoryBuffer:
    """
    A buffer for storing trajectories and computing advantages using Generalized Advantage Estimation (GAE).

    This class supports both continuous and discrete action spaces as well as numerical and image observations.
    It buffers observations, actions, rewards, values, and log probabilities, and computes discounted returns
    and advantages for a complete trajectory.

    Parameters
    ----------
    is_continuous_action_space : bool
        Flag indicating whether the action space is continuous.
    observation_shape : Tuple[int, ...]
        Shape of the observation space.
    num_actions : int
        Number of actions (or dimensionality for continuous actions).
    size : int
        Maximum number of transitions that can be stored in the buffer.
    discount : float
        Discount factor for future rewards.
    gae_lambda : float
        Lambda parameter for Generalized Advantage Estimation.
    """

    def __init__(
        self,
        is_continuous_action_space: bool,
        observation_shape: Tuple[int, ...],
        num_actions: int,
        size: int,
        discount: float,
        gae_lambda: float,
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
        self.discount: float = discount
        self.gae_lambda: float = gae_lambda
        self.pointer: int = 0
        self.trajectory_start_index: int = 0

    # Computes Generalized Advantage Estimation (GAE)
    def _discounted_cumulative_sums(self, sequence: np.ndarray, discount: float):
        """
        Compute discounted cumulative sums / Generalized Advantage Estimation
        (GAE) of a sequence using a linear filter.

        Parameters
        ----------
        sequence : np.ndarray
            Array of values for which the cumulative sums are computed.
        discount : float
            Discount factor to apply at each step. Must be between 0 and 1.

        Returns
        -------
        np.ndarray
            Array containing the discounted cumulative sums.

        Raises
        ------
        ValueError
            If discount is not a float between 0 and 1.
        """
        if not (0.0 <= discount <= 1.0):
            raise ValueError('"discount" must be a float between 0 and 1.')

        return scipy.signal.lfilter([1], [1, float(-discount)], sequence[::-1], axis=0)[::-1]

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        logprob: float,
    ) -> None:
        """
        Insert a new transition into the trajectory buffer.

        Parameters
        ----------
        observation : np.ndarray
            The observation from the environment.
        action : np.ndarray
            The action taken.
        reward : float
            The reward received after taking the action.
        value : float
            The estimated value of the current state.
        logprob : float
            The log probability of the taken action.

        Raises
        ------
        ValueError
            If the buffer is full and cannot accept more transitions.
        """
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
        """
        Finalize the current trajectory by computing advantages and returns.

        The method computes the temporal-difference residuals and applies the discounted cumulative
        sum to obtain the advantage estimates using Generalized Advantage Estimation (GAE), and computes
        the discounted returns.

        Parameters
        ----------
        last_value : float, optional
            The value estimate for the final state (default is 0).
        """
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.discount * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = self._discounted_cumulative_sums(
            deltas, self.discount * self.gae_lambda
        )
        self.return_buffer[path_slice] = self._discounted_cumulative_sums(rewards, self.discount)[
            :-1
        ]

        self.trajectory_start_index = self.pointer

    def get_and_reset(self, mini_batch_size: int) -> tf.data.Dataset:
        """
        Retrieve the stored trajectories and reset the buffer for reuse.

        This method creates a dataset from the stored arrays and applies per-batch
        normalization on the advantage estimates.

        Returns
        -------
        tf.data.Dataset
            A dataset yielding tuples containing:
            - observation_buffer: Array of observations.
            - action_buffer: Array of actions.
            - advantage_buffer: Per-batch normalized advantage estimates.
            - return_buffer: Discounted returns.
            - logprob_buffer: Log probabilities of actions.
            - value_buffer: Estimated state values.

        Raises
        ------
        ValueError
            If the buffer is not full when attempting to retrieve the trajectories.
        """
        if self.pointer != self.size:
            raise ValueError("get_and_empty called before buffer was full")

        # Do not normalize the entire advantage_buffer here.
        # Instead, simply reset the pointers and create the dataset with raw values.
        self.pointer, self.trajectory_start_index = 0, 0

        dataset = tf.data.Dataset.from_tensor_slices(
            (
                self.observation_buffer,
                self.action_buffer,
                self.advantage_buffer,
                self.return_buffer,
                self.logprob_buffer,
                self.value_buffer,
            )
        )

        dataset = dataset.shuffle(buffer_size=self.size)
        dataset = dataset.batch(mini_batch_size)

        # Normalizes advantages per mini-batch.
        def normalize_advantages(obs, actions, advantages, returns, logprobs, values):
            adv_mean = tf.reduce_mean(advantages)
            adv_std = tf.math.reduce_std(advantages)
            normalized_advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            return obs, actions, normalized_advantages, returns, logprobs, values

        dataset = dataset.map(normalize_advantages)

        return dataset
