import avocado_run
import gymnasium as gym
from keras.api.layers import Conv2D, Dense, Flatten, Input
from keras.api.models import Sequential
from keras.api.optimizers import Adam

from agents import DoubleDQNAgent, ModelTrainingConfig

if __name__ == "__main__":
    env = gym.make(id="AvocadoRun-v0")

    learning_rate = 0.001

    input_shape = env.observation_space.shape
    output_shape = env.action_space.n

    model: Sequential = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, kernel_size=3, activation="relu", padding="same"))
    model.add(Conv2D(64, kernel_size=3, activation="relu", padding="same"))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(units=output_shape, activation="linear"))
    model.compile(
        loss="mse",
        optimizer=Adam(learning_rate=learning_rate),  # type: ignore
    )

    agent = DoubleDQNAgent(env=env, obervation_normalization_type="image")

    # Based on best hyperparams found using Bayesian Hyperparameter Optimization - See wandb sweep sandy-sweep-16
    config: ModelTrainingConfig = {
        "env_name": "AvocadoRun",
        "project_name": "AvocadoRun",
        "replay_buffer_size": 50_000,
        "min_replay_buffer_size": 10_000,
        "minibatch_size": 64,
        "discount": 0.95,
        "training_frequency": 16,
        "update_target_every": 1000,
        "learning_rate": 0.001,
        "prop_steps_epsilon_decay": 0.9,
        "starting_epsilon": 1,
        "min_epsilon": 0.05,
        "steps_to_train": 200_000,
        "episode_metrics_window": 50,
    }

    agent.train(model=model, config=config, use_wandb=True)
