import gold_run_mini
import gymnasium as gym

from agents import DDQNAgent, DDQNModelTrainingConfig

from .grm_untrained_model import get_grm_untrained_model

# Params based on sweep 7877attm, run wise-sweep-15
config: DDQNModelTrainingConfig = {
    "env_name": "GoldRunMini",
    "project_name": "GoldRunMini_train",
    "replay_buffer_size": 75_000,
    "min_replay_buffer_size": 10_000,
    "minibatch_size": 128,
    "discount": 0.95,
    "training_frequency": 16,
    "update_target_every": 2_000,
    "learning_rate": 0.001,
    "prop_steps_epsilon_decay": 0.9,
    "starting_epsilon": 1,
    "min_epsilon": 0.05,
    "steps_to_train": 500_000,
    "episode_metrics_window": 100,
}

if __name__ == "__main__":
    env = gym.make(
        id="GoldRunMini-v1",
        render_mode=None,
        render_raw_pixels=False,
        disable_early_termination=False,
    )
    input_shape = env.observation_space.shape
    output_shape = env.action_space.n
    agent = DDQNAgent(env=env, obervation_normalization_type="image")
    model = get_grm_untrained_model(input_shape=input_shape, output_shape=output_shape)
    agent.train(model=model, config=config, save_model_every=20_000, use_wandb=False)
