from typing import Any, Dict

import wandb

from ..custom_types.model_training_config import ModelTrainingConfig
from .reward_queue import RewardQueue


class WandbLogger:
    """
    A logger class for integrating with Weights & Biases (wandb) for logging
    training metrics during reinforcement learning experiments.

    This class allows logging of step-level and episode-level metrics, including
    step count, epsilon value, episode rewards, rolling averages, and tumbling
    window statistics. It supports both standalone and sweep-based runs, ensuring
    flexibility in tracking training progress.

    Parameters
    ----------
    log_active : bool
        Whether logging is active.
    sweep_active : bool
        Whether the wandb sweep is active.
    config : ModelTrainingConfig or None
        Configuration object for wandb. Must include `project_name` if
        `sweep_active` is False.

    Raises
    ------
    ValueError
        If `log_active` is True, `sweep_active` is False, and `config` is None,
        a ValueError is raised to ensure that standalone runs have a valid configuration.

    Methods
    -------
    log_step_data(steps_passed, epsilon)
        Logs step-level data, including the current step count and epsilon value.
    log_episode_data(episodes_passed, reward_queue, episode_reward, episode_metrics_window)
        Logs episode-level data, including episode reward, rolling average reward,
        and tumbling window statistics. Also tracks min and max rewards within
        the tumbling window.
    """

    def __init__(
        self, log_active: bool, sweep_active: bool, config: ModelTrainingConfig | None
    ) -> None:
        self.log_active = log_active

        if not log_active:
            return

        if sweep_active:
            wandb.init()  # type: ignore
        elif not config:
            raise ValueError("If the run is not a sweep, the config cannot be 'None'")

        else:
            wandb.init(project=config["project_name"], config=config, mode="online")  # type: ignore

    def log_step_data(self, steps_passed: int, epsilon: float) -> None:
        """
        Logs step-level data such as the current step count and epsilon value.

        Parameters
        ----------
        steps_passed : int
            The number of steps that have passed.
        epsilon : float
            The current value of epsilon (e.g., exploration rate).
        """
        if not self.log_active:
            return
        wandb.log(  # type: ignore
            {
                "step": steps_passed,
                "epsilon": epsilon,
            }
        )

    def log_episode_data(
        self,
        episodes_passed: int,
        reward_queue: RewardQueue,
        episode_reward: float,
        episode_metrics_window: int,
    ) -> None:
        """
        Logs episode-level data such as episode reward, rolling average reward,
        and tumbling window metrics.

        Parameters
        ----------
        episodes_passed : int
            The number of episodes that have passed.
        reward_queue : RewardQueue
            An instance of the RewardQueue class, which tracks rewards for recent episodes.
        episode_reward : float
            The reward obtained in the current episode.
        episode_metrics_window : int
            The size of the window used for tumbling window metrics.
        """
        if not self.log_active:
            return

        log_data = {"episode": episodes_passed, "episode_reward": episode_reward}

        if reward_queue.get_size() < episode_metrics_window:
            wandb.log(log_data)  # type: ignore
            return

        average_reward = reward_queue.get_average_reward()
        log_data["rolling_window_average_reward"] = average_reward

        if episodes_passed % episode_metrics_window == 0:
            max_reward = reward_queue.get_max_reward()
            min_reward = reward_queue.get_min_reward()
            log_data["tumbling_window_average_reward"] = average_reward
            log_data["min_reward_in_tumbling_window"] = min_reward
            log_data["max_reward_in_tumbling_window"] = max_reward

        wandb.log(log_data)  # type: ignore
