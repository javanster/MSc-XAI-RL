import gymnasium as gym
import minecart_counter
import pygame

from agents import HumanPlayer

if __name__ == "__main__":
    env = gym.make(id="MinecartCounter-v2", render_mode="human", scatter_minecarts=True)

    key_action_mapping = {
        pygame.K_w: 0,
        pygame.K_e: 1,
        pygame.K_d: 2,
        pygame.K_c: 3,
        pygame.K_s: 4,
        pygame.K_z: 5,
        pygame.K_a: 6,
        pygame.K_q: 7,
    }

    human_player = HumanPlayer(env=env, key_action_mapping=key_action_mapping)

    human_player.play(default_action=8)
