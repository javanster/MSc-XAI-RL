import box_escape
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper

from agents import DDQNAgent, ModelTrainingConfig

from .be_untrained_models import get_be_untrained_conv_ff_model_v1

config: ModelTrainingConfig = {
    "env_name": "BoxEscape",
    "project_name": "BoxEscape",
    "model_architecture": "v1",
    "replay_buffer_size": 75_000,
    "min_replay_buffer_size": 10_000,
    "minibatch_size": 128,
    "discount": 0.95,
    "training_frequency": 8,
    "update_target_every": 1_000,
    "learning_rate": 0.0001,
    "prop_steps_epsilon_decay": 0.9,
    "starting_epsilon": 1,
    "min_epsilon": 0.01,
    "steps_to_train": 500_000,
    "episode_metrics_window": 50,
}

if __name__ == "__main__":
    env = gym.make(
        id="BoxEscape-v1",
        render_mode=None,
        fully_observable=True,
        curriculum_level=1,
    )
    env = ImgObsWrapper(RGBImgObsWrapper(env))
    agent = DDQNAgent(env=env, obervation_normalization_type="image")
    model = get_be_untrained_conv_ff_model_v1()
    agent.train(model=model, config=config, use_wandb=True)
