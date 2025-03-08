from typing import Tuple

import tensorflow as tf
from keras.api.initializers import RandomNormal
from keras.api.layers import Conv2D, Dense, Flatten, Input, Lambda
from keras.api.models import Model


def scale_inputs(x):
    x = tf.cast(x, tf.float32)
    return tf.divide(x, 255.0)


class CCRActorCritic(Model):
    def __init__(self, input_shape: Tuple[int, ...], num_actions: int):
        super().__init__()
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
        self.actor_mean = Dense(
            num_actions,
            activation="linear",
            kernel_initializer=RandomNormal(stddev=0.01),  # type: ignore
            name="actor_mean",
        )
        self.critic = Dense(1, activation="linear", kernel_initializer="orthogonal", name="critic")

        # Create custom variable via add_weight
        self.actor_logstd = self.add_weight(
            shape=(num_actions,),
            initializer=tf.zeros_initializer(),
            trainable=True,
            name="actor_logstd",
        )

        # Use the input_shape to build the model (weights will be created)
        self.build((None, *input_shape))

    def call(self, inputs):
        x = self.scaling(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        actor_out = self.actor_mean(x)
        critic_out = self.critic(x)
        return actor_out, critic_out
