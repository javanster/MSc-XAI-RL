from typing import TypedDict


class PPOModelTrainingConfig(TypedDict):
    """
    Configuration parameters for training a Proximal Policy Optimization (PPO) model.

    Attributes
    ----------
    env_name : str
        Name of the environment used for training.
    project_name : str
        Name of the project for logging and experiment tracking.
    training_steps : int
        Total number of training steps to execute.
    steps_per_epoch : int
        Number of steps taken per training epoch.
    learning_rate : float
        Learning rate used by the optimizer.
    discount : float
        Discount factor for future rewards.
    update_iterations : int
        Number of update iterations per epoch.
    mini_batch_size : int
        Size of the mini-batch used during training updates.
    gae_lambda : float
        Lambda parameter for Generalized Advantage Estimation (GAE).
    clip_ratio : float
        Clipping ratio for policy updates.
    target_kl : float
        Target Kullback-Leibler divergence threshold.
    ent_coef : float
        Coefficient for the entropy bonus term.
    vf_coef : float
        Coefficient for the value function loss.
    max_grad_norm : float
        Maximum norm for gradient clipping.
    use_rollback : bool
        Flag indicating whether to use rollback mechanisms.
    episode_metrics_window : int
        Number of episodes over which to compute rolling average metrics.
    """

    env_name: str
    project_name: str
    training_steps: int
    steps_per_epoch: int
    learning_rate: float
    discount: float
    update_iterations: int
    mini_batch_size: int
    gae_lambda: float
    clip_ratio: float
    target_kl: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    use_rollback: bool
    episode_metrics_window: int
