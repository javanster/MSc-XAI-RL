from typing import Any, Callable, Dict

import numpy as np


class TestLogger:
    """
    A logger class for logging information at each time step during testing of an agent.

    Parameters
    ----------
    log_vars : dict of {str: Any}
        Dictionary of variables to track and update.
    step_callback : Callable
        Function that processes each step and updates `log_vars`.
    """

    def __init__(
        self,
        log_vars: Dict[str, Any],
        step_callback: Callable[
            [np.ndarray, float, bool, bool, Dict[Any, Any], Dict[str, Any]], None
        ],
    ) -> None:

        self.log_vars = log_vars
        self.step_callback = step_callback

    def log_step(
        self,
        observation: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[Any, Any],
    ) -> None:
        """
        Logs a step by calling the step callback and updating log variables.

        Parameters
        ----------
        observation : np.ndarray
            The observed state at the current step.
        reward : float
            The reward received at the current step.
        terminated : bool
            Whether the episode has terminated (i.e., reached a terminal state).
        truncated : bool
            Whether the episode was truncated (ended early).
        info : dict of {Any: Any}
            Additional step-related information.

        Returns
        -------
        None
        """
        self.step_callback(observation, reward, terminated, truncated, info, self.log_vars)
