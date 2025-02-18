from collections import deque

import numpy as np
from gymnasium import Env, Wrapper
from gymnasium.spaces import Box


class FrameStack(Wrapper):
    """
    A gymnasium environment wrapper for the CCR env for stacking consecutive observations.

    This wrapper maintains a fixed-size deque of the most recent observations from the environment.
    On each call to step or reset, it returns a stacked observation created by concatenating the
    individual observations along the last axis.

    Parameters
    ----------
    env : gymnasium.Env
        The environment to be wrapped.
    k : int
        The number of consecutive frames to stack.

    Attributes
    ----------
    k : int
        The number of frames to stack.
    frames : collections.deque
        A deque storing the most recent observations with a maximum length of k.
    observation_space : gymnasium.spaces.Box
        The modified observation space after stacking k frames.
    """

    def __init__(self, env: Env, k: int) -> None:
        """
        Initialize the FrameStack wrapper.

        Parameters
        ----------
        env : gymnasium.Env
            The environment to be wrapped.
        k : int
            The number of consecutive frames to stack.
        """
        super(FrameStack, self).__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * k),
            dtype=env.observation_space.dtype,
        )

    def reset(self):
        """
        Reset the environment and initialize the frame stack.

        Returns
        -------
        observation : np.ndarray
            The stacked observation after reset, constructed by repeating the initial observation k times.
        info : dict
            An empty dictionary, reserved for additional reset information.
        """
        out = self.env.reset()
        if isinstance(out, tuple):
            ob = out[0]
        else:
            ob = out
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob(), {}

    def step(self, action: np.ndarray | int):
        """
        Execute an action in the environment and update the frame stack.

        Parameters
        ----------
        action : Any
            The action to be executed in the environment.

        Returns
        -------
        observation : np.ndarray
            The stacked observation after executing the action.
        reward : float
            The reward obtained after executing the action.
        terminated : bool
            Whether the episode has terminated.
        truncated : bool
            Whether the episode has been truncated.
        info : dict
            A dictionary containing additional information from the environment.
        """
        out = self.env.step(action)
        if isinstance(out[0], tuple):
            ob = out[0][0]
        else:
            ob = out[0]
        reward, terminated, truncated, info = out[1], out[2], out[3], out[4]
        self.frames.append(ob)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        """
        Concatenate the stored frames along the last axis to form a single observation.

        Returns
        -------
        np.ndarray
            The stacked observation array.
        """
        return np.concatenate(list(self.frames), axis=-1)
