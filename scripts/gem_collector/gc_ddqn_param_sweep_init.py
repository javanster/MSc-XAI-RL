import gem_collector
import gymnasium as gym

import wandb
from agents import DDQNAgent

from .gc_untrained_model import get_gc_untrained_model

sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "tumbling_window_average_reward"},
    "parameters": {
        "env_name": {"values": ["GemCollector"]},
        "replay_buffer_size": {"values": [25_000, 50_000, 75_000, 100_000, 150_000, 200_000]},
        "min_replay_buffer_size": {"values": [10_000]},
        "minibatch_size": {"values": [32, 64, 128]},
        "discount": {"values": [0.9, 0.95, 0.99]},
        "training_frequency": {"values": [i for i in range(4, 17)]},
        "update_target_every": {"values": [500, 750, 1000, 1250, 1500, 1750, 2_000]},
        "learning_rate": {"values": [0.0001, 0.001, 0.01]},
        "prop_steps_epsilon_decay": {"values": [0.9]},
        "starting_epsilon": {"values": [1]},
        "min_epsilon": {"values": [0.01, 0.05]},
        "steps_to_train": {"values": [500_000]},
        "episode_metrics_window": {"values": [100]},
    },
}


def train():
    env = gym.make(id="GemCollector-v3", render_mode="human")
    agent = DDQNAgent(env=env, obervation_normalization_type="image")
    model = get_gc_untrained_model()
    agent.train(config=None, model=model, use_wandb=True, use_sweep=True)


if __name__ == "__main__":
    wandb.login()  # type: ignore
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="GemCollector")  # type: ignore
    print(f"SWEEP ID: {sweep_id}")
    wandb.agent(sweep_id=sweep_id, function=train, count=100)  # type: ignore
