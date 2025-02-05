import gem_collector
import gymnasium as gym
import pygame

from agents import HumanPlayer

if __name__ == "__main__":
    env = gym.make(id="GemCollector-v3", render_mode="human", show_raw_pixels=False)

    key_action_mapping = {
        pygame.K_a: 0,
        pygame.K_d: 1,
        pygame.K_s: 2,
    }

    human_player = HumanPlayer(env=env, key_action_mapping=key_action_mapping)

    human_player.play(default_action=2)
