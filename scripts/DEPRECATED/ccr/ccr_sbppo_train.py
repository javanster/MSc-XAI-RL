import ccr
import gymnasium as gym
from ccr.wrappers import FrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

import wandb


def get_env():
    env = gym.make("CCR-v5")
    env = Monitor(env)  # Wrap the environment
    env = FrameStack(env=env, k=4)
    return env


if __name__ == "__main__":
    # Initialize W&B
    run = wandb.init(  # type: ignore
        project="CCR",
        config={
            "policy_type": "CnnPolicy",
            "total_timesteps": 10_000,
            "env_name": "CCR-v5",
            "n_envs": 8,
            "frame_stack": 4,
        },
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    # Create vectorized environment
    vec_env = make_vec_env(get_env, n_envs=wandb.config.n_envs, vec_env_cls=SubprocVecEnv)  # type: ignore

    # Initialize the PPO model
    model = PPO(
        wandb.config.policy_type,  # type: ignore
        vec_env,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
    )

    # Create evaluation environment
    eval_env = get_env()

    # Define the evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"ppo_models/CCR/{wandb.run.name}",  # type: ignore
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    # Define the W&B callback
    wandb_callback = WandbCallback(model_save_path=None, verbose=2)

    # Combine callbacks
    callback = CallbackList([eval_callback, wandb_callback])

    # Train the model
    model.learn(
        total_timesteps=wandb.config.total_timesteps,  # type: ignore
        callback=callback,
    )

    # Finish the W&B run
    run.finish()
