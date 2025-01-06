from typing import TypedDict


class ModelTrainingConfig(TypedDict):
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
