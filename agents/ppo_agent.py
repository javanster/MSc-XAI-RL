import os
from typing import Tuple

import ccr

os.environ["KERAS_BACKEND"] = "tensorflow"

import gymnasium as gym
import keras
import numpy as np
import tensorflow as tf
from gymnasium import Env
from keras.api.layers import Conv2D, Dense, Flatten, Input
from keras.api.models import Sequential
from keras.api.optimizers import Adam
from keras.api.random import SeedGenerator, categorical

from .util_classes.trajectory_buffer import TrajectoryBuffer


class PPOAgent:

    def __init__(
        self,
        env: Env,
        observation_shape,
        image_observations: bool,
        actor_model: Sequential,
        critic_model: Sequential,
        steps_per_epoch: int,
        policy_learning_rate: float = 3e-4,
        value_function_learning_rate: float = 1e-3,
    ) -> None:
        self.env: Env = env
        self.continuous_action_space = isinstance(env.action_space, gym.spaces.Box)
        self.num_actions: int = (
            env.action_space.shape[0] if self.continuous_action_space else env.action_space.n
        )

        self.trajectory_buffer = TrajectoryBuffer(
            is_continuous_action_space=self.continuous_action_space,
            observation_shape=observation_shape,
            num_actions=self.num_actions,
            size=steps_per_epoch,
        )
        self.image_observations = image_observations

        self.seed_generator: SeedGenerator = SeedGenerator(1337)
        self.policy_optimizer: Adam = Adam(learning_rate=policy_learning_rate)
        self.value_optimizer: Adam = Adam(learning_rate=value_function_learning_rate)

        self.steps_per_epoch: int = steps_per_epoch
        self.gamma: float = 0.99
        self.clip_ratio: float = 0.2
        self.train_policy_iterations: int = 50
        self.train_value_iterations: int = 50
        self.lam: float = 0.97
        self.target_kl: float = 0.01

        self._model_validation(actor_model=actor_model, critic_model=critic_model)
        self.actor: Sequential = actor_model
        self.critic: Sequential = critic_model

    def _model_validation(self, actor_model: Sequential, critic_model: Sequential) -> None:
        """Validates actor and critic models for both discrete and continuous action spaces."""

        # Validate Critic Model
        critic_output_units = critic_model.layers[-1].units
        if critic_output_units != 1:
            raise ValueError(
                "The critic model must have exactly one output unit (shape should be (batch_size, 1))."
            )

        # Validate Actor Model
        if self.continuous_action_space:
            # Continuous action spaces should output means and log stds for each action dim
            expected_units = self.num_actions * 2  # Means + log std
            actor_output_units = actor_model.layers[-1].units

            if actor_output_units != expected_units:
                raise ValueError(
                    f"Continuous action space detected, but actor model output has {actor_output_units} units."
                    f" Expected {expected_units} (num_actions * 2)."
                )

            actor_activation_func = actor_model.layers[-1].activation.__name__
            if actor_activation_func not in ["linear", "tanh"]:
                raise ValueError(
                    "For continuous actions, the actor model's last layer must have 'tanh' or 'linear' activation."
                )

        else:
            # Discrete action space - Expect logits for categorical sampling
            actor_activation_func = actor_model.layers[-1].activation.__name__
            if actor_activation_func not in ["linear", "softmax"]:
                raise ValueError(
                    "For discrete actions, the actor model's last layer should use 'linear' or 'softmax' activation."
                )

        # Remove unnecessary compilation attributes
        actor_model.optimizer = None
        actor_model.compiled_loss = None
        critic_model.optimizer = None
        critic_model.compiled_loss = None

    def _logprobabilities(self, logits: tf.Tensor, a: tf.Tensor) -> tf.Tensor:
        """
        Computes the log-probabilities of taking actions 'a' under the policy defined by 'logits'.
        Supports both discrete and continuous action spaces.

        Parameters:
        - logits: The raw output of the actor network.
        - a: The action(s) taken.

        Returns:
        - logprobability: The log probability of the selected action(s).
        """

        if self.continuous_action_space:
            # Continuous case: Log probability of sampled action under Gaussian policy

            # Ensure logits has the correct shape
            tf.ensure_shape(logits, [None, self.num_actions * 2])

            # Split logits into means and log standard deviations
            means, log_stds = tf.split(logits, num_or_size_splits=2, axis=-1)  # type: ignore
            stds = tf.exp(log_stds)  # Convert log standard deviations to standard deviations

            # Compute log probability using Gaussian log likelihood formula
            logprob = -0.5 * (((a - means) / stds) ** 2 + 2 * log_stds + tf.math.log(2 * np.pi))

            # Sum over action dimensions since each action dimension is independent
            logprobability = tf.reduce_sum(logprob, axis=-1)

        else:
            # Discrete case: Compute log probabilities using categorical distribution
            logprobabilities_all = tf.math.log_softmax(logits, axis=1)
            actions_one_hot = tf.one_hot(a, depth=self.num_actions, dtype=tf.float32)
            logprobability = tf.reduce_sum(actions_one_hot * logprobabilities_all, axis=1)

        return logprobability

    @tf.function
    def _sample_action(self, observation: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Samples an action based on the policy output. Supports both discrete and continuous spaces."""

        # Normalize image observations
        if self.image_observations:
            observation = tf.divide(observation, 255)

        # Get logits from the actor network
        logits = self.actor(observation)
        logits = tf.convert_to_tensor(logits, dtype=tf.float32)  # Ensure logits is a Tensor

        if self.continuous_action_space:
            tf.ensure_shape(logits, [None, self.num_actions * 2])

            # Split logits into means and log standard deviations
            means, log_stds = tf.split(logits, num_or_size_splits=2, axis=-1)  # type: ignore
            stds = tf.exp(log_stds)

            # Sample action using Gaussian noise
            noise = tf.random.normal(shape=tf.shape(means), mean=0.0, stddev=1.0)
            action = means + stds * noise

            # Clip actions within valid range
            low = tf.convert_to_tensor(self.env.action_space.low, dtype=tf.float32)
            high = tf.convert_to_tensor(self.env.action_space.high, dtype=tf.float32)
            action = tf.clip_by_value(action, low, high)

        else:
            action = categorical(logits, 1, seed=self.seed_generator)
            action = tf.squeeze(action, axis=1)  # Remove extra dimension
            action = tf.cast(action, tf.int32)  # Ensure action is an integer

        return logits, tf.squeeze(action, axis=0)  # Remove batch dimension before returning

    # Train the policy by maxizing the PPO-Clip objective
    @tf.function
    def _train_policy(
        self, observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
    ):

        if self.continuous_action_space:
            # Reshape from (batch_size, flattened_dim) → (batch_size, height, width, channels)
            height, width, channels = self.env.observation_space.shape
            observation_buffer = tf.reshape(observation_buffer, (-1, height, width, channels))

        if self.image_observations:
            observation_buffer = tf.divide(observation_buffer, 255)

        with tf.GradientTape() as tape:
            # Compute current log probabilities
            logits = self.actor(observation_buffer)
            current_logprobabilities = self._logprobabilities(
                logits, action_buffer
            )  # Shape: (batch_size,)

            # Calculate ratio (pi_theta / pi_theta_old)
            ratio = tf.exp(current_logprobabilities - logprobability_buffer)  # Shape: (batch_size,)

            # Calculate clipped advantage
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            )  # Shape: (batch_size,)

            # Calculate policy loss
            clipped_advantage = tf.minimum(
                ratio * advantage_buffer, min_advantage
            )  # Shape: (batch_size,)
            policy_loss = -tf.reduce_mean(clipped_advantage)

        # Compute gradients and update actor network
        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        assert isinstance(policy_grads, list), "policy_grads should be a list."
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

        # Calculate KL divergence for diagnostics
        kl_div = logprobability_buffer - current_logprobabilities  # Shape: (batch_size,)
        kl_div_mean = tf.reduce_mean(kl_div)
        kl_div_sum = tf.reduce_sum(kl_div_mean)  # Scalar

        return kl_div_sum

    # Train the value function by regression on mean-squared error
    @tf.function
    def _train_value_function(self, observation_buffer, return_buffer):
        if self.continuous_action_space:
            # Reshape from (batch_size, flattened_dim) → (batch_size, height, width, channels)
            height, width, channels = self.env.observation_space.shape
            observation_buffer = tf.reshape(observation_buffer, (-1, height, width, channels))

        if self.image_observations:
            observation_buffer = tf.divide(observation_buffer, 255)  # Normalize to [0,1]

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = keras.ops.mean((return_buffer - self.critic(observation_buffer)) ** 2)
        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        assert isinstance(value_grads, list), "value_grads should be a list."
        self.value_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))

    def _reset_env_and_tracking_vars(self) -> Tuple[np.ndarray, int, int]:
        observation, _ = self.env.reset()
        episode_reward_return: int = 0
        episode_step_length: int = 0
        return observation, episode_reward_return, episode_step_length

    def train(self, env: Env, epochs: int):
        self.env = env
        observation, episode_reward_return, episode_step_length = (
            self._reset_env_and_tracking_vars()
        )

        for epoch in range(epochs):
            # Initialize the sum of the returns, lengths and number of episodes for each epoch
            sum_return = 0
            sum_length = 0
            num_episodes = 0

            # Iterate over the steps of each epoch
            for t in range(1, self.steps_per_epoch + 1):

                # Get the logits, action, and take one step in the environment
                if self.continuous_action_space:
                    observation = tf.expand_dims(observation, axis=0)
                else:
                    observation = observation.reshape(1, -1)

                logits, action = self._sample_action(observation)  # type: ignore
                if self.continuous_action_space:
                    env_action = (
                        action.numpy().squeeze()
                    )  # Ensures correct shape (3,) for continuous spaces
                else:
                    env_action = int(
                        action.numpy()
                    )  # Converts tensor to integer for discrete spaces

                next_observation, reward, terminated, truncated, _ = env.step(env_action)

                episode_reward_return += reward
                episode_step_length += 1

                obs_normalized = observation / 255

                value_t = self.critic(
                    obs_normalized
                )  # Estimation of how much reward may be collected in the future from the state?
                logprobability_t = self._logprobabilities(
                    logits, action
                )  # How probable the chosen action is under the current policy, given the observation?

                # Store obs, act, rew, v_t, logp_pi_t
                self.trajectory_buffer.insert(
                    observation.numpy() if isinstance(observation, tf.Tensor) else observation,  # type: ignore
                    action.numpy() if isinstance(action, tf.Tensor) else action,  # type: ignore
                    float(reward),  # No change needed here (reward is already a scalar)
                    float(
                        value_t.numpy().item()  # type: ignore
                        if isinstance(value_t, tf.Tensor)
                        else value_t.item() if isinstance(value_t, np.ndarray) else value_t
                    ),
                    float(
                        logprobability_t.numpy().item()  # type: ignore
                        if isinstance(logprobability_t, tf.Tensor)
                        else (
                            logprobability_t.item()
                            if isinstance(logprobability_t, np.ndarray)
                            else logprobability_t
                        )
                    ),
                )

                # Update the observation
                observation = next_observation

                # Finish trajectory if reached to a terminal state
                if terminated or truncated or (t == self.steps_per_epoch - 1):
                    if truncated or (t == self.steps_per_epoch - 1):
                        if self.continuous_action_space:
                            observation = tf.expand_dims(observation, axis=0)
                        else:
                            observation = observation.reshape(1, -1)

                        if self.image_observations:
                            observation = observation / 255

                        last_value = self.critic(observation)

                    else:
                        last_value = 0
                    self.trajectory_buffer.complete_episode_trajectory(last_value)
                    sum_return += episode_reward_return
                    sum_length += episode_step_length
                    num_episodes += 1
                    observation, episode_reward_return, episode_step_length = (
                        self._reset_env_and_tracking_vars()
                    )

            # Get values from the buffer
            (
                observation_buffer,
                action_buffer,
                advantage_buffer,
                return_buffer,
                logprobability_buffer,
            ) = self.trajectory_buffer.get_and_reset()

            # Update the policy and implement early stopping using KL divergence
            for _ in range(self.train_policy_iterations):
                kl = self._train_policy(
                    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
                )
                if not isinstance(kl, tf.Tensor):
                    raise ValueError("kl was not a Tensor")
                if kl > 1.5 * self.target_kl:
                    # Early Stopping
                    break

            # Update the value function
            for l in range(self.train_value_iterations):
                self._train_value_function(observation_buffer, return_buffer)

            # Print mean return and length for each epoch
            print(
                f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
            )

    def test(self, env, render=True):
        """Runs the trained PPO agent in the given environment."""
        self.env = env
        observation, _ = self.env.reset()
        terminated = False

        while not terminated:
            # Reshape observation correctly
            if self.continuous_action_space:
                observation = tf.expand_dims(observation, axis=0)  # Add batch dimension
            else:
                observation = observation.reshape(1, -1)  # Keep as a vector for MLPs

            # Get policy output
            if self.image_observations:
                observation = observation / 255  # Normalize image before feeding it to the network
            logits = self.actor(observation)

            if self.continuous_action_space:
                # Continuous actions: Extract means from the policy output (ignore log_std)
                means, _ = tf.split(logits, num_or_size_splits=2, axis=-1)  # type: ignore
                action = means  # In test mode, use deterministic actions (no sampling)
            else:
                # Discrete actions: Choose the action with the highest probability
                action = tf.argmax(logits, axis=1)

            # Convert action to NumPy format and step in the environment
            env_action = action.numpy().squeeze() if self.continuous_action_space else int(action)

            observation, _, terminated, _, _ = env.step(env_action)

            if render:
                env.render()


if __name__ == "__main__":

    ENV = "not"

    if ENV == "car":

        env = gym.make("CCR-v5")

        input_shape = env.observation_space.shape
        actor_output_shape = env.action_space.shape[0]

        actor_model = Sequential()
        actor_model.add(Input(shape=input_shape))
        actor_model.add(Conv2D(32, kernel_size=3, activation="relu", padding="same"))
        actor_model.add(Conv2D(64, kernel_size=3, activation="relu", padding="same"))
        actor_model.add(Flatten())
        actor_model.add(Dense(64, activation="relu"))
        actor_model.add(Dense(64, activation="relu"))
        actor_model.add(Dense(units=actor_output_shape * 2, activation="tanh"))

        critic_model = Sequential()
        critic_model.add(Input(shape=input_shape))
        critic_model.add(Conv2D(32, kernel_size=3, activation="relu", padding="same"))
        critic_model.add(Conv2D(64, kernel_size=3, activation="relu", padding="same"))
        critic_model.add(Flatten())
        critic_model.add(Dense(64, activation="relu"))
        critic_model.add(Dense(64, activation="relu"))
        critic_model.add(Dense(units=1, activation="linear"))

        agent = PPOAgent(
            observation_shape=input_shape,
            image_observations=True,
            actor_model=actor_model,
            critic_model=critic_model,
            env=env,
            steps_per_epoch=5,
        )

        agent.train(env=env, epochs=1)

        env = gym.make("CCR-v5", render_mode="human")

        agent.test(env=env)

    else:

        env = gym.make("CartPole-v1")
        observation_dimensions = env.observation_space.shape[0]

        input_shape = env.observation_space.shape
        output_shape = env.action_space.n

        actor_model = Sequential()
        actor_model.add(Input(shape=input_shape))
        actor_model.add(Dense(64, activation="tanh"))
        actor_model.add(Dense(64, activation="tanh"))
        actor_model.add(Dense(units=output_shape, activation=None))

        critic_model = Sequential()
        critic_model.add(Input(shape=input_shape))
        critic_model.add(Dense(64, activation="tanh"))
        critic_model.add(Dense(64, activation="tanh"))
        critic_model.add(Dense(units=1, activation="linear"))

        agent = PPOAgent(
            observation_shape=input_shape,
            image_observations=False,
            actor_model=actor_model,
            critic_model=critic_model,
            env=env,
            steps_per_epoch=4000,
        )

        agent.train(env=env, epochs=5)

        env = gym.make("CartPole-v1", render_mode="human")

        agent.test(env=env)
