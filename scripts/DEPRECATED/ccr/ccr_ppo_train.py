import ccr
import gymnasium as gym
from ccr.wrappers import FrameStack

from agents import PPOAgent, PPOModelTrainingConfig

from .ccr_actor_critic_model import CCRActorCritic

if __name__ == "__main__":
    env = gym.make("CCR-v5")
    env = FrameStack(env=env, k=4)
    input_shape = env.observation_space.shape
    print(f"INPUT SHAPE: {input_shape}")
    num_actions = env.action_space.shape[
        0
    ]  # For a 3-dimensional continuous action space (or adjust for discrete)

    # Create the shared actor-critic model
    shared_model = CCRActorCritic(input_shape=input_shape, num_actions=num_actions)

    # Create the PPO agent with the shared model and optional actor_logstd
    agent = PPOAgent(
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
        observation_shape=input_shape,
        config=config,
        shared_model=shared_model,
        use_wandb=False,
        use_sweep=False,
    )
