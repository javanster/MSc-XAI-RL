import gymnasium as gym

import wandb
from agents import FrameStack, PPOAgent

from .ccr_actor_critic_model import get_ccr_actor_logstd, get_ccr_untrained_model

sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "tumbling_window_average_reward"},
    "parameters": {
        "env_name": {"value": "CCR"},
        "project_name": {"value": "CCR"},
        "steps_per_epoch": {"distribution": "int_uniform", "min": 1024, "max": 4096},
        "learning_rate": {"distribution": "log_uniform_values", "min": 5e-5, "max": 5e-4},
        "use_rollback": {"values": [True, False]},
        "gae_lambda": {"distribution": "uniform", "min": 0.95, "max": 0.98},
        "clip_ratio": {"distribution": "uniform", "min": 0.1, "max": 0.3},
        "discount": {"value": 0.99},
        "ent_coef": {"distribution": "log_uniform_values", "min": 1e-4, "max": 0.01},
        "max_grad_norm": {"distribution": "uniform", "min": 0.3, "max": 1.0},
        "target_kl": {"distribution": "uniform", "min": 0.008, "max": 0.02},
        "update_iterations": {"distribution": "int_uniform", "min": 4, "max": 40},
        "mini_batch_size": {"values": [32, 64, 96, 128, 192, 256]},
        "vf_coef": {"distribution": "uniform", "min": 0.2, "max": 0.8},
        "training_steps": {"value": 501_760},
        "episode_metrics_window": {"value": 5},
    },
}


def train():
    env = gym.make("CCR-v5")
    env = FrameStack(env=env, k=4)
    agent = PPOAgent(env=env)
    input_shape = env.observation_space.shape
    num_actions = env.action_space.shape[0]
    actor_logstd = (
        get_ccr_actor_logstd(num_actions=num_actions)
        if isinstance(env.action_space, gym.spaces.Box)
        else None
    )
    model = get_ccr_untrained_model(input_shape=input_shape, num_actions=num_actions)
    agent.train(
        env=env,
        observation_shape=input_shape,
        config=None,
        shared_model=model,
        actor_logstd=actor_logstd,
        use_wandb=True,
        use_sweep=True,
    )


if __name__ == "__main__":
    wandb.login()  # type: ignore
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="CCR")  # type: ignore
    print(f"SWEEP ID: {sweep_id}")
    wandb.agent(sweep_id=sweep_id, function=train, count=100)  # type: ignore
