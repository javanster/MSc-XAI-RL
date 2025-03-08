import gold_run_mini
import gymnasium as gym
import pygame

from agents import HumanPlayer

if __name__ == "__main__":
    env = gym.make(
        id="GoldRunMini-v1",
        render_mode="human",
        render_raw_pixels=True,
        disable_early_termination=False,
        no_lava_termination=False,
        only_second_room=False,
        lava_spots=8,
        render_fps=3,
    )

    key_action_mapping = {
        pygame.K_w: 0,
        pygame.K_d: 1,
        pygame.K_s: 2,
        pygame.K_a: 3,
    }

    human_player = HumanPlayer(env=env, key_action_mapping=key_action_mapping)

    human_player.play(default_action=4)
