from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper

from agents import DDQNAgent, ModelTrainingConfig

from .bt_untrained_model import get_bt_untrained_model
from .test import BoxToggle

# Based on best hyperparams found using Bayesian Hyperparameter Optimization - See wandb sweep comic-sweep-18
config: ModelTrainingConfig = {
    "env_name": "BoxToggle",
    "project_name": "BoxToggle",
    "replay_buffer_size": 75_000,
    "min_replay_buffer_size": 10_000,
    "minibatch_size": 128,
    "discount": 0.95,
    "training_frequency": 6,
    "update_target_every": 1_750,
    "learning_rate": 0.0001,
    "prop_steps_epsilon_decay": 0.9,
    "starting_epsilon": 1,
    "min_epsilon": 0.01,
    "steps_to_train": 150_000,
    "episode_metrics_window": 50,
}

if __name__ == "__main__":
    env = BoxToggle()
    env = ImgObsWrapper(RGBImgObsWrapper(env))
    agent = DDQNAgent(env=env, obervation_normalization_type="image")
    model = get_bt_untrained_model()
    agent.train(model=model, config=config, use_wandb=True)
