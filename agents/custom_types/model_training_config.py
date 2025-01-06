from typing import TypedDict


class ModelTrainingConfig(TypedDict):
    """
    A TypedDict defining the required configuration for training a reinforcement learning model.

    This class specifies the structure and types of the configuration dictionary, ensuring that all
    necessary fields are present and correctly typed for use in training.

    Attributes
    ----------
    env_name : str
        The name of the environment in which the agent will train.
    project_name : str
        The name of the project for organizational purposes. Used by Wandb if active.
    replay_buffer_size : int
        The maximum size of the replay buffer for storing past experiences.
    min_replay_buffer_size : int
        The minimum number of experiences required in the replay buffer before training starts.
    minibatch_size : int
        The number of experiences to sample for each training update.
    discount : float
        The discount factor (gamma) used to calculate target Q-values.
    training_frequency : int
        The number of steps between training updates for the online model.
    update_target_every : int
        The number of steps between updates of the target model's weights.
    learning_rate : float
        The learning rate for the optimizer used during training.
    prop_steps_epsilon_decay : float
        The proportion of total training steps over which epsilon will decay.
    starting_epsilon : float
        The initial value of epsilon for epsilon-greedy action selection.
    min_epsilon : float
        The minimum allowable value for epsilon.
    steps_to_train : int
        The total number of steps to execute during training.
    episode_metrics_window : int
        The number of episodes over which to calculate rolling metrics such as average reward.
    """

    env_name: str
    project_name: str
    replay_buffer_size: int
    min_replay_buffer_size: int
    minibatch_size: int
    discount: float
    training_frequency: int
    update_target_every: int
    learning_rate: float
    prop_steps_epsilon_decay: float
    starting_epsilon: float
    min_epsilon: float
    steps_to_train: int
    episode_metrics_window: int
