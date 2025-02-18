import os
import time
from typing import Tuple

import ccr

os.environ["KERAS_BACKEND"] = "tensorflow"

import gymnasium as gym
import keras
import numpy as np
import tensorflow as tf
from gymnasium import Env
from keras.api.layers import Dense, Flatten, Input
from keras.api.models import Model
from keras.api.optimizers import Adam
from keras.api.random import SeedGenerator
from tqdm import tqdm

import wandb

from .custom_types.ppo_model_training_config import PPOModelTrainingConfig
from .util_classes.reward_queue import RewardQueue
from .util_classes.trajectory_buffer import TrajectoryBuffer
from .util_classes.wandb_logger import WandbLogger


class PPOAgentModelClass:

    def __init__(
        self,
        env: Env,
    ) -> None:

        self.env: Env = env
        self.is_continuous_action_space: bool = isinstance(env.action_space, gym.spaces.Box)
        self.num_actions: int = (
            env.action_space.shape[0] if self.is_continuous_action_space else env.action_space.n
        )

        self.seed_generator: SeedGenerator = SeedGenerator(28)

    def _logprobabilities(self, logits: tf.Tensor, a: tf.Tensor) -> tf.Tensor:
        if self.is_continuous_action_space:
            # For continuous actions, logits is now just the actor mean.
            tf.ensure_shape(logits, [None, self.num_actions])
            stds = tf.exp(self.actor_logstd)  # Compute standard deviation from actor_logstd
            log_stds = self.actor_logstd
            # Compute Gaussian log likelihood elementwise:
            logprob = -0.5 * (((a - logits) / stds) ** 2 + 2 * log_stds + tf.math.log(2 * np.pi))  # type: ignore
            logprobability = tf.reduce_sum(logprob, axis=-1)
        else:
            tf.ensure_shape(logits, [None, self.num_actions])
            actions_one_hot = tf.one_hot(a, depth=self.num_actions, dtype=tf.float32)
            logprobabilities_all = tf.math.log_softmax(logits, axis=1)
            logprobability = tf.reduce_sum(actions_one_hot * logprobabilities_all, axis=1)
        return logprobability

    @tf.function
    def _sample_action(self, observation: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Get the shared outputs: for continuous, actor_out is the mean; for discrete, actor_out are logits.
        actor_out, _ = self.shared_model(observation)

        if self.is_continuous_action_space:
            std = tf.exp(self.actor_logstd)
            noise = tf.random.normal(shape=tf.shape(actor_out), mean=0.0, stddev=1.0)
            action = actor_out + std * noise
            # Clip actions to the valid range
            low = tf.convert_to_tensor(self.env.action_space.low, dtype=tf.float32)
            high = tf.convert_to_tensor(self.env.action_space.high, dtype=tf.float32)
            action = tf.clip_by_value(action, low, high)
        else:
            # For discrete actions, actor_out represents logits.
            action = tf.random.categorical(actor_out, num_samples=1)
            action = tf.squeeze(action, axis=1)
            action = tf.cast(action, tf.int32)
        return actor_out, tf.squeeze(action, axis=0)

    @tf.function
    def _train_minibatch(
        self, obs_batch, act_batch, adv_batch, ret_batch, logp_old_batch, old_value_batch
    ):
        """
        Perform one gradient update on the combined loss (policy + value loss) for a minibatch.
        """
        if self.is_continuous_action_space:
            height, width, channels = self.env.observation_space.shape
            obs_batch = tf.reshape(obs_batch, (-1, height, width, channels))
        # (Assuming that if using image observations, scaling is done in the model)

        # Ensure proper shapes for returns and old values
        ret_batch = tf.reshape(ret_batch, (-1, 1))
        old_value_batch = tf.reshape(old_value_batch, (-1, 1))

        with tf.GradientTape() as tape:
            # Forward pass: get actor output and value estimates
            actor_out, new_values = self.shared_model(obs_batch)
            new_values = tf.reshape(new_values, (-1, 1))

            # Compute log probabilities for the actions
            if self.is_continuous_action_space:
                std = tf.exp(self.actor_logstd)
                log_std = self.actor_logstd
                # Gaussian log likelihood
                logprob = -0.5 * (((act_batch - actor_out) / std) ** 2 + 2 * log_std + tf.math.log(2 * np.pi))  # type: ignore
                new_logp = tf.reduce_sum(logprob, axis=-1)
            else:
                new_logp = self._logprobabilities(actor_out, act_batch)

            # Ratio for PPO clipping
            ratio = tf.exp(new_logp - logp_old_batch)
            min_adv = tf.where(
                adv_batch > 0, (1 + self.clip_ratio) * adv_batch, (1 - self.clip_ratio) * adv_batch
            )
            clipped_adv = tf.minimum(ratio * adv_batch, min_adv)
            policy_loss = -tf.reduce_mean(clipped_adv)

            # Entropy bonus
            if self.is_continuous_action_space:
                entropy = tf.reduce_sum(
                    0.5 * (1.0 + tf.math.log(2 * np.pi)) + self.actor_logstd, axis=-1
                )
            else:
                probs = tf.nn.softmax(actor_out, axis=-1)
                entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1)  # type: ignore
            entropy_bonus = tf.reduce_mean(entropy)

            # Value loss (clipped as in PPO)
            new_values = tf.reshape(new_values, (-1,))
            old_values = tf.reshape(old_value_batch, (-1,))
            returns = tf.reshape(ret_batch, (-1,))
            v_loss_unclipped = tf.square(new_values - returns)
            v_clipped = old_values + tf.clip_by_value(
                new_values - old_values, -self.clip_ratio, self.clip_ratio
            )
            v_loss_clipped = tf.square(v_clipped - returns)
            value_loss = 0.5 * tf.reduce_mean(tf.maximum(v_loss_unclipped, v_loss_clipped))

            # Combined loss (note: PyTorch code subtracts the entropy bonus)
            total_loss = policy_loss - self.ent_coef * entropy_bonus + self.vf_coef * value_loss

        # Combine the trainable variables (including actor_logstd if continuous)
        trainable_vars = self.shared_model.trainable_variables
        if self.is_continuous_action_space:
            trainable_vars = trainable_vars + [self.actor_logstd]

        grads = tape.gradient(total_loss, trainable_vars)
        # Clip gradients as in PyTorch
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(clipped_grads, trainable_vars))

        # Also compute an approximate KL divergence for monitoring
        kl_div = logp_old_batch - new_logp
        avg_kl = tf.reduce_mean(kl_div)
        return avg_kl, total_loss

    def train(
        self,
        env: Env,
        observation_shape: Tuple[int, ...],
        config: PPOModelTrainingConfig | None,
        shared_model: Model,
        use_wandb: bool = False,
        use_sweep: bool = False,
    ):
        wandb_logger = WandbLogger(log_active=use_wandb, sweep_active=use_sweep, config=config)
        if use_wandb:
            config = wandb.config  # type: ignore

        if config is None:
            raise ValueError(
                "No active config. Either no config was given as an argument, or, if 'use_wandb' was given as True, wandb failed to retreive the config."
            )

        self.env = env
        self.best_tumbling_window_average: float = float("-inf")

        if self.is_continuous_action_space:
            if not hasattr(shared_model, "actor_logstd"):
                raise ValueError("The shared_model does not have 'actor_logstd' attribute.")
            self.actor_logstd = shared_model.actor_logstd

        self.use_rollback = config["use_rollback"]
        self.initial_lr = config["learning_rate"]  # Use one LR for all parameters
        self.optimizer = Adam(learning_rate=config["learning_rate"])
        self.vf_coef = config["vf_coef"]  # For example, matching the PyTorch default
        self.steps_per_epoch: int = config["steps_per_epoch"]
        self.clip_ratio: float = config["clip_ratio"]
        self.update_iterations: int = config["update_iterations"]
        self.gae_lambda: float = config["gae_lambda"]
        self.target_kl: float = config["target_kl"]
        self.ent_coef: float = config["ent_coef"]
        self.max_grad_norm: float = config["max_grad_norm"]
        self.mini_batch_size: int = config["mini_batch_size"]
        episode_metrics_window: int = config["episode_metrics_window"]
        env_name: str = config["env_name"]

        self.shared_model: Model = shared_model
        self.trajectory_buffer: TrajectoryBuffer = TrajectoryBuffer(
            is_continuous_action_space=self.is_continuous_action_space,
            observation_shape=observation_shape,
            num_actions=self.num_actions,
            size=config["steps_per_epoch"],
            discount=config["discount"],
            gae_lambda=config["gae_lambda"],
        )

        reward_queue = RewardQueue(maxlen=episode_metrics_window)

        observation, _ = self.env.reset()
        episode_reward = 0
        model_dir = self._ensure_and_get_model_dir_path(env_name=env_name, use_wandb=use_wandb)
        num_episodes = 0
        num_steps_total_steps = 0
        average_reward = None
        max_reward = None
        min_reward = None

        epochs = config["training_steps"] // self.steps_per_epoch
        for epoch in tqdm(range(epochs), unit="epoch", total=epochs):
            # Compute fraction for annealing (linearly decaying here)
            frac = 1.0 - (
                epoch / epochs
            )  # total_epochs computed as training_steps // steps_per_epoch
            new_lr = self.initial_lr * frac
            # Update the optimizer's learning rate:
            self.optimizer.learning_rate.assign(new_lr)  # type: ignore
            print(f"Epoch {epoch+1}: Learning Rate = {new_lr:.6f}")

            # Iterate over the steps of each epoch
            for t in range(1, self.steps_per_epoch + 1):

                # Get the logits, action, and take one step in the environment
                if self.is_continuous_action_space:
                    observation = tf.expand_dims(observation, axis=0)
                else:
                    observation = observation.reshape(1, -1)

                logits, action = self._sample_action(observation)  # type: ignore
                if self.is_continuous_action_space:
                    env_action = (
                        action.numpy().squeeze()
                    )  # Ensures correct shape (3,) for continuous spaces
                else:
                    env_action = int(
                        action.numpy()
                    )  # Converts tensor to integer for discrete spaces

                next_observation, reward, terminated, truncated, _ = env.step(env_action)

                episode_reward += reward

                # obs_normalized = observation / 255

                _, value_t = self.shared_model(
                    observation
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
                        if self.is_continuous_action_space:
                            observation = tf.expand_dims(observation, axis=0)
                        else:
                            observation = observation.reshape(1, -1)

                        _, last_value = self.shared_model(observation)

                    else:
                        last_value = 0
                    self.trajectory_buffer.complete_episode_trajectory(last_value)

                    num_episodes += 1
                    reward_queue.update(reward=episode_reward)
                    wandb_logger.log_episode_data(
                        episodes_passed=num_episodes,
                        episode_metrics_window=episode_metrics_window,
                        episode_reward=episode_reward,
                        reward_queue=reward_queue,
                    )
                    observation, _ = self.env.reset()
                    episode_reward = 0

                    if (
                        reward_queue.get_size() >= episode_metrics_window
                        and num_episodes % episode_metrics_window == 0
                    ):
                        average_reward = reward_queue.get_average_reward()

                        # Checks every tumbling window episode whether a new best static average is reached
                        if average_reward > self.best_tumbling_window_average:
                            self.best_tumbling_window_average = average_reward

                            max_reward = reward_queue.get_max_reward()
                            min_reward = reward_queue.get_min_reward()

                            self._save_model(
                                model_dir=model_dir,
                                average_reward=average_reward,
                                max_reward=max_reward,
                                min_reward=min_reward,
                            )

                num_steps_total_steps += 1
                wandb_logger.log_step_data(steps_passed=num_steps_total_steps, epsilon=None)

            wandb_logger.log_epoch_data(learning_rate=new_lr)

            # Get values from the buffer
            (
                observation_buffer,
                action_buffer,
                advantage_buffer,
                return_buffer,
                logprobability_buffer,
                old_value_buffer,
            ) = self.trajectory_buffer.get_and_reset()

            # Create a tf.data.Dataset from the collected data
            dataset = tf.data.Dataset.from_tensor_slices(
                (
                    observation_buffer,
                    action_buffer,
                    advantage_buffer,
                    return_buffer,
                    logprobability_buffer,
                    old_value_buffer,
                )
            )

            # Shuffle the dataset. Here we use the total number of steps as the buffer size.
            dataset = dataset.shuffle(buffer_size=self.steps_per_epoch)

            # Define your mini-batch size (you can set this as a hyperparameter)
            dataset = dataset.batch(self.mini_batch_size)

            # --- Policy Update with Mini-Batches ---
            # Save a backup of model weights for possible rollback:
            backup_weights = self.shared_model.get_weights()
            if self.is_continuous_action_space:
                backup_logstd = self.actor_logstd.numpy().copy()  # type: ignore

            for update_epoch in range(self.update_iterations):
                total_kl = 0.0
                total_loss_sum = 0.0
                num_batches = 0
                for (
                    obs_batch,
                    act_batch,
                    adv_batch,
                    ret_batch,
                    logp_old_batch,
                    old_value_batch,
                ) in dataset:  # type: ignore
                    kl, loss = self._train_minibatch(
                        obs_batch, act_batch, adv_batch, ret_batch, logp_old_batch, old_value_batch
                    )  # type: ignore
                    total_kl += kl  # type: ignore
                    total_loss_sum += loss
                    num_batches += 1
                avg_kl = total_kl / num_batches
                avg_loss = total_loss_sum / num_batches
                print(f"Average KL divergence at update epoch {update_epoch+1}: {avg_kl:.6f}")
                print(f"Average loss at update epoch {update_epoch+1}: {avg_loss:.6f}")

                wandb_logger.log_training_epoch_data(avg_loss=avg_loss, avg_kl=avg_kl)

                # Early stopping and optional rollback check
                if avg_kl > 1.5 * self.target_kl or avg_kl < -1.5 * self.target_kl:
                    if self.use_rollback:
                        print(
                            f"Rollback triggered at update epoch {update_epoch+1} due to KL divergence."
                        )
                        self.shared_model.set_weights(backup_weights)
                        if self.is_continuous_action_space:
                            self.actor_logstd.assign(backup_logstd)  # type: ignore
                    else:
                        print(
                            f"Early stopping triggered at update epoch {update_epoch+1} due to KL divergence."
                        )
                    break

                # If average loss is too large
                if avg_loss > 1000.0:
                    print(
                        f"Rollback triggered at update epoch {update_epoch+1} due to large average loss."
                    )
                    self.shared_model.set_weights(backup_weights)
                    if self.is_continuous_action_space:
                        self.actor_logstd.assign(backup_logstd)  # type: ignore

                    break

        self.env.close()

        # Save final model
        if average_reward and max_reward and min_reward:
            self._save_model(
                model_dir=model_dir,
                average_reward=average_reward,
                max_reward=max_reward,
                min_reward=min_reward,
            )

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
        self, model_dir: str, average_reward: float, max_reward: float, min_reward: float
    ) -> None:

        model_file_path = f"{model_dir}/{int(time.time())}_model_{average_reward:_>9.4f}avg_{max_reward:_>9.4f}max_{min_reward:_>9.4f}min.keras"
        self.shared_model.save(model_file_path)

    def test(self, env, model: Model, render=True):
        self.shared_model = model
        self.env = env
        observation, _ = self.env.reset()
        terminated = False

        while not terminated:
            if self.is_continuous_action_space:
                observation = tf.expand_dims(observation, axis=0)
            else:
                observation = observation.reshape(1, -1)

            actor_out, _ = self.shared_model(observation)
            if self.is_continuous_action_space:
                # For continuous actions, use the actor mean directly (deterministic)
                action = actor_out
            else:
                # For discrete actions, take the argmax of logits
                action = tf.argmax(actor_out, axis=1)
            env_action = (
                action.numpy().squeeze() if self.is_continuous_action_space else int(action)
            )
            observation, _, terminated, _, _ = env.step(env_action)
            if render:
                env.render()


def create_mlp_actor_critic_model(input_shape, num_actions):
    """Creates a simple MLP-based actor-critic model for CartPole."""
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = Dense(64, activation="tanh", kernel_initializer="orthogonal")(x)
    x = Dense(64, activation="tanh", kernel_initializer="orthogonal")(x)
    # Actor head: outputs logits for each discrete action.
    actor_logits = Dense(
        num_actions,
        activation="linear",
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),  # type: ignore
        name="actor_logits",
    )(x)
    # Critic head: outputs a scalar value.
    critic = Dense(1, activation="linear", kernel_initializer="orthogonal", name="critic")(x)
    return Model(inputs=inputs, outputs=[actor_logits, critic])


""" if __name__ == "__main__":
    # Create the CartPole environment.
    env = gym.make("CartPole-v1")
    input_shape = env.observation_space.shape  # For CartPole, typically (4,)
    num_actions = env.action_space.n  # For CartPole, there are 2 discrete actions.

    # Create the shared actor-critic model.
    shared_model = create_mlp_actor_critic_model(input_shape, num_actions)

    # For discrete action spaces, actor_logstd is not used.
    actor_logstd = None

    # Create the PPO agent.
    # Note: steps_per_epoch here refers to the number of transitions collected per update.
    # Set use_rollback=True if you want rollback on KL overshoot, or False for plain early stopping.
    agent = PPOAgent(
        env=env,
        observation_shape=input_shape,
        shared_model=shared_model,
        steps_per_epoch=4000,
        learning_rate=3e-3,
        use_rollback=False,  # Change to False if you only want early stopping.
        actor_logstd=actor_logstd,
    )

    # Train for 8000 steps (adjust as needed)
    agent.train(env=env, training_steps=120_000)

    # After training, test the agent with rendering enabled.
    env = gym.make("CartPole-v1", render_mode="human")
    agent.test(env=env) """


if __name__ == "__main__":
    pass

"""self.shared_model: Model = shared_model
        self.actor_logstd: tf.Variable | None = actor_logstd
        self.initial_lr = learning_rate  # Use one LR for all parameters
        self.optimizer = Adam(learning_rate=learning_rate)
        self.vf_coef = 0.5  # For example, matching the PyTorch default

        self.steps_per_epoch: int = steps_per_epoch
        self.discount: float = 0.99
        self.clip_ratio: float = 0.2
        self.update_iterations: int = 80
        self.train_value_iterations: int = 80
        self.gae_lambda: float = 0.97
        self.target_kl: float = 0.01
        self.ent_coef: float = 0.01
        self.max_grad_norm: float = 0.5"""
