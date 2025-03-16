import gem_collector
import gymnasium as gym
import numpy as np
import pygame

from .ensure_save_dir_exists import ensure_save_directory_exists


def gc_save_human_obs_from_rgb_array_obs(
    array_observations: np.ndarray, save_dir: str, file_prefix: str
) -> None:
    """
    Render and save human-readable images from an array of observations.

    This function iterates over a numpy array of observations, renders each observation using the GemCollector environment,
    and saves the resulting image as a PNG file in the specified directory with a given file prefix.

    Parameters
    ----------
    array_observations : np.ndarray
        An array where each element represents the grid state observation to be rendered.
    save_dir : str
        The directory path where the rendered images will be saved.
    file_prefix : str
        The prefix used for the filenames of the saved images.
    """
    pygame.init()
    env = gym.make(id="GemCollector-v3", render_mode=None, show_raw_pixels=False, render_fps=1000)
    env = env.unwrapped
    env.reset()

    if env.window is None:
        env.window = pygame.display.set_mode((env.window_size, env.window_size))

    for i, obs in enumerate(array_observations):
        env.set_state_based_on_obs_grid(obs)
        env._draw_entities()
        ensure_save_directory_exists(save_dir)
        pygame.image.save(env.window, f"{save_dir}{file_prefix}_human_obs_{i}.png")
    pygame.quit()
