import ccr
import gymnasium as gym
import numpy as np
from gymnasium.utils.play import play

if __name__ == "__main__":

    step_n: int = 0

    def update_step(obs_t, obs_tp1, action, rew, terminated, truncated, info):
        global step_n
        step_n += 1
        print(step_n)

    play(
        gym.make("CCR-v5", render_mode="rgb_array"),
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
        callback=update_step,
    )
