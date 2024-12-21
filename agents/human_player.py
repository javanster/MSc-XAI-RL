import time
from typing import Any, Dict

import pygame
from gymnasium import Env


class HumanPlayer:
    """
    A class that enables a human player to control an agent in Gymnasium environments using a keyboard.

    This class facilitates interaction with Gymnasium environments by mapping keyboard inputs
    to predefined agent actions. The player can control the agent during episodes using key presses,
    and a default action is performed if no input is received within a specified timeout period.

    Parameters
    ----------
    env : gymnasium.Env
        The Gymnasium environment where the human player will interact as the agent.
    key_action_mapping : dict
        A dictionary mapping keyboard keys to agent actions.
        For example, `{pygame.K_w: 0, pygame.K_s: 1}` maps "W" and "S" keys to actions 0 and 1.

    Attributes
    ----------
    env : gymnasium.Env
        The Gymnasium environment instance.
    key_action_mapping : dict
        A dictionary defining the mapping of keyboard keys to agent actions.

    Methods
    -------
    play(default_action, timeout=25, episodes=10)
        Starts an interactive session allowing the human player to control the agent.
    """

    def __init__(self, env: Env, key_action_mapping: Dict[Any, int]):
        self.env = env
        self.key_action_mapping = key_action_mapping

    def _get_human_action(self):
        """
        Maps key presses to agent actions.
        Returns the action corresponding to the key pressed by the user, provided by the given
        key-action mapping function.
        """
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                return self.key_action_mapping.get(event.key)
            elif event.type == pygame.QUIT:
                self.env.close()
        return None

    def play(
        self,
        default_action: int,
        timeout: int = 30,
        episodes: int = 10,
    ):
        """
        Allows a human player to control the agent via the keyboard in the Gymnasium environment.

        The function facilitates interactive control where the agent performs actions based on
        user input from the keyboard. If no key is pressed within the specified timeout period,
        the agent performs a default action. The interaction spans over a given number of episodes.

        Parameters
        ----------
        default_action : int
            The action to perform if no key press is detected within the timeout period.
        timeout : int, optional
            The maximum time (in milliseconds) to wait for a key press before defaulting
            to the default action. Default is 25 ms.
        episodes : int, optional
            The number of episodes for which the human player can interact with the environment.
            Default is 10.

        Raises
        ------
        EnvironmentError
            If the environment cannot be rendered in "human" mode.
        pygame.error
            If there is an issue initializing or handling pygame events.

        Examples
        --------
        >>> key_action_mapping = {pygame.K_w: 0, pygame.K_d: 1, pygame.K_s: 2, pygame.K_a: 3}
        >>> human_player = HumanPlayer(env, key_action_mapping)
        >>> human_player.play(default_action=4, timeout=50, episodes=5)
        """
        for _ in range(episodes):

            self.env.reset()

            self.env.render()

            terminated = False

            while not terminated:
                action = None
                start_ticks = pygame.time.get_ticks()

                while action is None:
                    action = self._get_human_action()

                    if pygame.time.get_ticks() - start_ticks > timeout:
                        action = default_action

                _, _, terminated, _, _ = self.env.step(action=action)

                self.env.render()

            time.sleep(2)

        self.env.close()
