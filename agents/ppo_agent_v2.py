import datetime
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, Union

os.environ["KERAS_BACKEND"] = "tensorflow"

import gymnasium as gym
import numpy as np
import tensorflow as tf
from keras.api.models import Model
from keras.api.optimizers import Adam
from keras.api.random import SeedGenerator
from tqdm import tqdm

import wandb

from .custom_types.ppo_model_training_config import PPOModelTrainingConfig
from .util_classes.reward_queue import RewardQueue
from .util_classes.wandb_logger import WandbLogger


class CustomBeta:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample(self):
        # Sample from two Gamma distributions and normalize.
        sample_x = tf.random.gamma(shape=tf.shape(self.alpha), alpha=self.alpha, dtype=tf.float32)
        sample_y = tf.random.gamma(shape=tf.shape(self.beta), alpha=self.beta, dtype=tf.float32)
        return sample_x / (sample_x + sample_y)

    def log_prob(self, x):
        # Calculate the log probability density using the Beta PDF formula:
        # log_pdf = lgamma(alpha+beta) - lgamma(alpha) - lgamma(beta)
        #           + (alpha - 1)*log(x) + (beta - 1)*log(1-x)
        log_norm = (
            tf.math.lgamma(self.alpha + self.beta)
            - tf.math.lgamma(self.alpha)
            - tf.math.lgamma(self.beta)
        )
        return (
            log_norm
            + (self.alpha - 1.0) * tf.math.log(x)
            + (self.beta - 1.0) * tf.math.log(1.0 - x)
        )


class PPOV2Agent:
    """
    WORK IN PROGRESS
    """

    def __init__(self, env: gym.Env) -> None:
        self.env: gym.Env = env
        self.is_continuous_action_space: bool = isinstance(env.action_space, gym.spaces.Box)
        self.num_actions: int = (
            env.action_space.shape[0] if self.is_continuous_action_space else env.action_space.n
        )
        self.seed_generator: SeedGenerator = SeedGenerator(28)
        self.shared_model: Optional[Model] = None
        # For logging episodic rewards (using the same RewardQueue as before)
        self.reward_queue = None

    def get_action(self, observation: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
        # Expand dims if needed
        obs_batch = np.expand_dims(observation, axis=0)
        beta_params, _ = self.shared_model(obs_batch)  # type: ignore
        beta_params = beta_params[0]  # shape (num_actions, 2)
        alpha = beta_params[:, 0]
        beta = beta_params[:, 1]
        distribution = CustomBeta(alpha, beta)
        action = distribution.sample()
        log_prob = tf.reduce_sum(distribution.log_prob(action))
        # Rescale action from [0, 1] to env.action_space
        low = np.array(self.env.action_space.low).flatten()
        high = np.array(self.env.action_space.high).flatten()
        action_np = action.numpy()  # should be shape (3,)
        action_rescaled = low + (high - low) * action_np
        # Ensure a flat 1-D numpy array of type float32
        action_rescaled = np.array(action_rescaled, dtype=np.float32).flatten()
        return action_rescaled, log_prob  # type: ignore

    def _ensure_and_get_model_dir_path(self, env_name: str, use_wandb: bool) -> str:
        if not os.path.exists("ppo_models"):
            os.makedirs("ppo_models")

        model_dir: str = (
            f"ppo_models/{env_name}/{wandb.run.name}"  # type: ignore
            if use_wandb
            else f"ppo_models/{env_name}/run_{int(time.time())}"
        )

        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    def _save_model(
        self,
        model_dir: str,
        episode_n: int,
        reward_queue: RewardQueue,
    ) -> None:
        average_reward = reward_queue.get_average_reward()
        max_reward = reward_queue.get_max_reward()
        min_reward = reward_queue.get_min_reward()

        model_file_path = f"{model_dir}/model_ep_{episode_n}_{average_reward:_>9.4f}avg_{max_reward:_>9.4f}max_{min_reward:_>9.4f}min.keras"
        self.shared_model.save(model_file_path)  # type: ignore

    def train(
        self,
        env: gym.Env,
        config: PPOModelTrainingConfig | None,
        shared_model: Model,
        use_wandb: bool = False,
        use_sweep: bool = False,
    ):
        """
        This training loop follows the structure of the second code example:
          • Roll out entire episodes, storing transitions in a buffer.
          • When the buffer is full, perform several PPO update epochs over mini-batches.
          • The actor head outputs Beta distribution parameters.
          • Logging and model saving follow the original code.
        """
        wandb_logger = WandbLogger(log_active=use_wandb, sweep_active=use_sweep, config=config)
        if use_wandb:
            config = wandb.config  # type: ignore

        if config is None:
            raise ValueError("No active config provided.")

        self.env = env
        self.shared_model = shared_model
        self.shared_model.compile(optimizer=Adam(learning_rate=config["learning_rate"]))  # type: ignore
        model_dir = self._ensure_and_get_model_dir_path(
            env_name=config["env_name"], use_wandb=use_wandb
        )

        # Use the same RewardQueue from the original code for episodic logging
        self.reward_queue = RewardQueue(maxlen=config["episode_metrics_window"])

        # Training hyperparameters (from the config and adapted from the second code)
        episodes = config.get("episodes", 1000)
        buffer_size = config.get("buffer_size", 2000)
        batch_size = config.get("mini_batch_size", 128)
        gamma = config["discount"]
        ppo_epochs = config.get("update_iterations", 10)
        clip_epsilon = config.get("clip_ratio", 0.1)

        # Buffer for transitions: each element is a tuple:
        # (observation, action, log_prob, reward, next_observation, value)
        transitions = []
        episode_rewards = []

        for episode in tqdm(range(1, episodes + 1), unit="episode"):
            observation, _ = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Sample action using the Beta-based policy
                action, log_prob = self.get_action(observation)
                # Step the environment (assumes continuous action space)
                new_observation, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward

                # Get critic value for the current state
                # (We assume the shared_model returns (beta_params, critic_value))
                _, value = self.shared_model(np.expand_dims(observation, axis=0))
                value = tf.squeeze(value).numpy()
                transitions.append((observation, action, log_prob, reward, new_observation, value))

                observation = new_observation

                # When the buffer is full, perform PPO update
                if len(transitions) >= buffer_size:
                    # Prepare tensors from transitions
                    states = tf.convert_to_tensor([t[0] for t in transitions], dtype=tf.float32)
                    actions = tf.convert_to_tensor([t[1] for t in transitions], dtype=tf.float32)
                    old_log_probs = tf.convert_to_tensor(
                        [t[2] for t in transitions], dtype=tf.float32
                    )
                    rewards_tensor = tf.convert_to_tensor(
                        [t[3] for t in transitions], dtype=tf.float32
                    )
                    # For bootstrapping, get critic values for next_states:
                    next_states = tf.convert_to_tensor(
                        [t[4] for t in transitions], dtype=tf.float32
                    )
                    _, next_values = self.shared_model(next_states)
                    next_values = tf.squeeze(next_values, axis=1)
                    # Get stored critic values
                    values = tf.convert_to_tensor([t[5] for t in transitions], dtype=tf.float32)

                    # Compute “discounted rewards” (one-step bootstrapped targets)
                    discounted_rewards = rewards_tensor + gamma * next_values
                    # Advantages: (target - value)
                    advantages = discounted_rewards - values
                    # (Optionally, you might normalize advantages here.)

                    # Define a simple generator for mini-batches
                    def gen_batches(indices, batch_size):
                        for i in range(0, len(indices), batch_size):
                            yield indices[i : i + batch_size]

                    indices = np.arange(buffer_size)
                    for _ in range(ppo_epochs):
                        np.random.shuffle(indices)
                        for batch in gen_batches(indices, batch_size):
                            batch_states = tf.gather(states, batch)
                            batch_actions = tf.gather(actions, batch)
                            batch_old_log_probs = tf.gather(old_log_probs, batch)
                            batch_advantages = tf.gather(advantages, batch)
                            batch_discounted_rewards = tf.gather(discounted_rewards, batch)

                            with tf.GradientTape() as tape:
                                # Forward pass: get new Beta parameters and critic values
                                beta_params, v_pred = self.shared_model(batch_states)
                                # Assume beta_params has shape (batch, num_actions, 2)
                                # Extract alpha and beta for each action dimension:
                                alpha = beta_params[..., 0]
                                beta = beta_params[..., 1]
                                dist = CustomBeta(alpha, beta)
                                new_log_probs = tf.reduce_sum(dist.log_prob(batch_actions), axis=1)
                                # PPO ratio (note: old_log_probs were stored as scalars)
                                ratio = tf.exp(new_log_probs - batch_old_log_probs)
                                surr1 = ratio * batch_advantages
                                surr2 = (
                                    tf.clip_by_value(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
                                    * batch_advantages
                                )
                                action_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                                # Value (critic) loss
                                v_pred = tf.squeeze(v_pred, axis=1)
                                value_loss = tf.reduce_mean(
                                    tf.square(batch_discounted_rewards - v_pred)
                                )
                                loss = action_loss + 2.0 * value_loss

                            grads = tape.gradient(loss, self.shared_model.trainable_variables)
                            # Optionally clip gradients here if desired:
                            self.shared_model.optimizer.apply_gradients(
                                zip(grads, self.shared_model.trainable_variables)  # type: ignore
                            )
                    transitions.clear()

            # End of episode: update reward queue and log episode data
            self.reward_queue.update(episode_reward)
            episode_rewards.append(episode_reward)
            wandb_logger.log_episode_data(
                episodes_passed=episode,
                reward_queue=self.reward_queue,
                episode_reward=episode_reward,
                episode_metrics_window=config["episode_metrics_window"],
            )
            wandb_logger.log_step_data(steps_passed=episode, epsilon=None)

            # Save model every 30 episodes and when a new best tumbling window average is reached
            if (
                self.reward_queue.get_size() >= config["episode_metrics_window"]
                and episode % config["episode_metrics_window"] == 0
            ):
                avg_reward = self.reward_queue.get_average_reward()
                if (
                    not hasattr(self, "best_tumbling_window_average")
                    or avg_reward > self.best_tumbling_window_average
                ):
                    self.best_tumbling_window_average = avg_reward
                    self._save_model(
                        model_dir=model_dir, episode_n=episode, reward_queue=self.reward_queue
                    )

            if episode % 30 == 0:
                self._save_model(
                    model_dir=model_dir, episode_n=episode, reward_queue=self.reward_queue
                )

        self.env.close()
        # Save final model
        self._save_model(model_dir=model_dir, episode_n=episodes, reward_queue=self.reward_queue)

    def test(self, env, model: Model, render=True):
        self.shared_model = model
        self.env = env
        observation, _ = self.env.reset()
        terminated = False

        while not terminated:
            obs = np.expand_dims(observation, axis=0)
            beta_params, _ = self.shared_model(obs)
            beta_params = beta_params[0]
            alpha = beta_params[:, 0]
            beta = beta_params[:, 1]
            dist = CustomBeta(alpha, beta)
            # For testing choose the mode (deterministic)
            action = (alpha - 1) / (alpha + beta - 2)
            low = self.env.action_space.low
            high = self.env.action_space.high
            action_rescaled = low + (high - low) * action.numpy()
            observation, _, terminated, _, _ = self.env.step(action_rescaled)
            if render:
                self.env.render()
