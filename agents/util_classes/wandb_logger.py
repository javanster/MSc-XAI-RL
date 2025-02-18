from typing import Any, Dict, Optional

import wandb

from ..custom_types.ddqn_model_training_config import DDQNModelTrainingConfig
from ..custom_types.ppo_model_training_config import PPOModelTrainingConfig
from .reward_queue import RewardQueue


class WandbLogger:
    """
    Logger for integrating with Weights & Biases (wandb) to track reinforcement learning training metrics.

    This class supports logging of metrics at different granularities including steps, episodes, and epochs.
    It handles both standalone and sweep-based runs. In standalone runs, a configuration object containing
    a 'project_name' is required.

    Parameters
    ----------
    log_active : bool
        Flag indicating whether logging is enabled.
    sweep_active : bool
        Flag indicating whether the wandb sweep mode is active.
    config : DDQNModelTrainingConfig or PPOModelTrainingConfig or None
        Configuration dictionary for wandb initialization. For standalone runs (i.e., when
        sweep_active is False), config must not be None and must include a 'project_name' key.

    Raises
    ------
    ValueError
        If logging is active, the run is not a sweep, and config is None.
    """

    def __init__(
        self,
        log_active: bool,
        sweep_active: bool,
        config: DDQNModelTrainingConfig | PPOModelTrainingConfig | None,
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

    def log_step_data(self, steps_passed: int, epsilon: Optional[float]) -> None:
        """
        Log step-level metrics including the current step count and epsilon value.

        Parameters
        ----------
        steps_passed : int
            The total number of steps completed so far.
        epsilon : float or None
            The current value of epsilon (exploration rate). Can be None if not applicable.
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
        Log episode-level metrics including episode reward and window-based statistics.

        This method logs the reward for the current episode and, when enough episodes have passed to fill
        the specified metrics window, logs the rolling average reward along with the minimum and maximum rewards
        observed in the window.

        Parameters
        ----------
        episodes_passed : int
            The total number of episodes completed so far.
        reward_queue : RewardQueue
            Instance of RewardQueue used to compute window-based reward statistics.
        episode_reward : float
            The reward obtained in the current episode.
        episode_metrics_window : int
            The number of episodes constituting the tumbling window for computing statistics.
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

    def log_epoch_data(self, learning_rate):
        """
        Log metrics for the current epoch, specifically the learning rate.

        Parameters
        ----------
        learning_rate : float
            The learning rate used in the current training epoch.
        """
        if not self.log_active:
            return

        wandb.log(  # type: ignore
            {
                "learning_rate": learning_rate,
            }
        )

    def log_training_epoch_data(self, avg_loss, avg_kl):
        """
        Log aggregated training metrics for the training epoch, including average loss and average KL divergence.

        Parameters
        ----------
        avg_loss : float
            The average loss computed over the training epoch.
        avg_kl : float
            The average Kullback-Leibler divergence computed over the training epoch.
        """

        if not self.log_active:
            return

        wandb.log(  # type: ignore
            {
                "avg_loss": avg_loss,
                "avg_kl": avg_kl,
            }
        )
