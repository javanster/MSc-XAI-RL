import gold_run_mini
import gymnasium as gym
import numpy as np
import pygame

from utils import ensure_directory_exists


def grm_save_human_obs_from_rgb_array_obs(
    array_observations: np.ndarray, save_dir: str, file_prefix: str, render_raw_pixels: bool
) -> None:
    """
    Render and save human-readable images from an array of observations.

    This function iterates over a numpy array of observations, renders each observation using the GoldRunMini environment,
    and saves the resulting image as a PNG file in the specified directory with a given file prefix.

    Parameters
    ----------
    array_observations : np.ndarray
        An array where each element represents the grid state observation to be rendered.
    save_dir : str
        The directory path where the rendered images will be saved.
    file_prefix : str
        The prefix used for the filenames of the saved images.
    render_raw_pixels : bool
        Whether the images should be saved as raw pixels or the human friendly version.
    """
    pygame.init()
    env = gym.make(
        id="GoldRunMini-v1", render_mode=None, render_fps=1000, render_raw_pixels=render_raw_pixels
    )
    env = env.unwrapped
    env.reset()

    if env.window is None:
        env.window = pygame.display.set_mode((env.window_size, env.window_size))

    for i, obs in enumerate(array_observations):
        env.set_state_based_on_obs_grid(obs)
        env._draw_entities()
        ensure_directory_exists(save_dir)
        pygame.image.save(env.window, f"{save_dir}{file_prefix}_human_obs_{i}.png")
    pygame.quit()
