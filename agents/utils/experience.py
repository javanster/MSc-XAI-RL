from typing import NamedTuple

import numpy as np


class Experience(NamedTuple):
    """
    A named tuple representing a single experience in a reinforcement learning environment.

    Each experience consists of the current state, the action taken, the reward received,
    the resulting new state, and whether the episode has terminated.

    Attributes
    ----------
    current_state : np.ndarray
        The state of the environment before the action was taken.
    action : int
        The action taken by the agent in the current state.
    reward : float
        The reward received after taking the action.
    new_state : np.ndarray
        The state of the environment after the action was taken.
    terminated : bool
        A boolean flag indicating whether the episode has terminated after this step.
    """

    current_state: np.ndarray
    action: int
    reward: float
    new_state: np.ndarray
    terminated: bool
