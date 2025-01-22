import gymnasium as gym
import numpy as np
from gymnasium.utils.play import play

if __name__ == "__main__":
    play(
        gym.make("CarRacing-v3", render_mode="rgb_array"),
        keys_to_action={
            "w": np.array([0, 0.7, 0], dtype=np.float32),
            "a": np.array([-1, 0, 0], dtype=np.float32),
            "s": np.array([0, 0, 1], dtype=np.float32),
            "d": np.array([1, 0, 0], dtype=np.float32),
            "wa": np.array([-1, 0.7, 0], dtype=np.float32),
            "dw": np.array([1, 0.7, 0], dtype=np.float32),
            "ds": np.array([1, 0, 1], dtype=np.float32),
            "as": np.array([-1, 0, 1], dtype=np.float32),
        },
        noop=np.array([0, 0, 0], dtype=np.float32),
    )
