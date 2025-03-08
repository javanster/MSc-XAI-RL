from typing import Tuple

import ccr
import gymnasium as gym
import tensorflow as tf
from ccr.wrappers import FrameStack
from keras.api.initializers import RandomNormal
from keras.api.layers import Conv2D, Dense, Flatten, Lambda
from keras.api.models import Model

from agents import PPOModelTrainingConfig, PPOV2Agent


def scale_inputs(x):
    x = tf.cast(x, tf.float32)
    return tf.divide(x, 255.0)


class CCRActorCritic(Model):
    def __init__(self, input_shape: Tuple[int, ...], num_actions: int):
        print(f"num_actions: {num_actions}")
        super().__init__()
        self.num_actions = num_actions  # Store it here
        self.scaling = Lambda(scale_inputs)
        self.conv1 = Conv2D(
            32, kernel_size=8, strides=4, activation="relu", kernel_initializer="orthogonal"
        )
        self.conv2 = Conv2D(
            64, kernel_size=4, strides=2, activation="relu", kernel_initializer="orthogonal"
        )
        self.conv3 = Conv2D(
            64, kernel_size=3, strides=1, activation="relu", kernel_initializer="orthogonal"
        )
        self.flatten = Flatten()
        self.dense = Dense(512, activation="relu", kernel_initializer="orthogonal")
        # Actor head now outputs 2 * num_actions units (for alpha and beta)
        self.actor_dense = Dense(
            num_actions * 2,
            activation="softplus",  # softplus to ensure positive outputs
            kernel_initializer=RandomNormal(stddev=0.01),  # type: ignore
            name="actor_dense",
        )
        self.critic = Dense(1, activation="linear", kernel_initializer="orthogonal", name="critic")

        # Build the model to create weights
        self.build((None, *input_shape))

    def call(self, inputs):
        x = self.scaling(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        # Actor: obtain parameters and reshape to (batch, num_actions, 2)
        actor_out = self.actor_dense(x)
        # Now use the stored number of actions instead of tf.shape
        beta_params = tf.reshape(actor_out, (-1, self.num_actions, 2)) + 1.0
        critic_out = self.critic(x)
        return beta_params, critic_out


if __name__ == "__main__":
    env = gym.make("CCR-v5")
    env = FrameStack(env=env, k=4)
    input_shape = env.observation_space.shape
    num_actions = env.action_space.shape[
        0
    ]  # For a 3-dimensional continuous action space (or adjust for discrete)

    # Create the shared actor-critic model
    shared_model = CCRActorCritic(input_shape=input_shape, num_actions=num_actions)

    # Create the PPO agent with the shared model and optional actor_logstd
    agent = PPOV2Agent(
        env=env,
    )

    config: PPOModelTrainingConfig = {
        "env_name": "CCR",
        "project_name": "CCR",
        "steps_per_epoch": 200,
        "learning_rate": 0.00005,
        "use_rollback": False,
        "gae_lambda": 0.97,
        "clip_ratio": 0.2,
        "discount": 0.97,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,
        "target_kl": 0.01,
        "update_iterations": 40,
        "mini_batch_size": 128,
        "vf_coef": 0.5,
        "training_steps": 501_760,
        "episode_metrics_window": 5,
    }

    agent.train(
        env=env,
        config=config,
        shared_model=shared_model,
        use_wandb=True,
        use_sweep=False,
    )
