from typing import Any, Dict, LiteralString, Optional, Tuple

import numpy as np
import pygame
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

from .entity import Entity


class AvocadoRun(Env):
    """
    A custom Gymnasium environment where the goal of the agent is to avoid moving enemies (angry farmers that
    partly move towards the agent), and eat the avocado.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: Optional[LiteralString] = None,
        render_fps: int = 4,
        num_avocados: int = 1,
        num_enemies: int = 2,
        aggressive_enemies: bool = False,
    ) -> None:
        super().__init__()
        self.name = "avocado_run"
        self.STEP_PENALTY = -0.0035
        self.ENEMY_HIT_PENALTY = -0.3
        self.AVOCADO_REWARD = 1.0035
        self.reward_range = (-1, 1)

        self.AVOCADO_COLOR = (0, 255, 0)
        self.AGENT_COLOR = (78, 172, 248)
        self.ENEMY_COLOR = (255, 0, 0)

        self.render_fps = render_fps
        self.num_avocados = num_avocados
        self.num_enemies = num_enemies
        self.aggressive_enemies = aggressive_enemies
        self.grid_side_length = 10
        self.action_space = Discrete(5)
        self.action_dict = {
            0: "up",
            1: "right",
            2: "down",
            3: "left",
            4: "do_nothing",
        }
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.grid_side_length, self.grid_side_length, 3),
            dtype=np.uint8,
        )
        self.window_size = 600
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        agent_starting_position: Optional[Tuple[int, int]] = None,
        avocado_starting_positions: Optional[Tuple[Tuple[int, int], ...]] = None,
        enemy_starting_positions: Optional[Tuple[Tuple[int, int], ...]] = None,
    ) -> Tuple[np.ndarray, Dict[Any, Any]]:
        super().reset(seed=seed)

        if avocado_starting_positions and len(avocado_starting_positions) != self.num_avocados:
            raise ValueError(
                """Number of elements in avocado_starting_positions must be equal to num_avocados
                provided upon class initilaization."""
            )

        if enemy_starting_positions and len(enemy_starting_positions) != self.num_enemies:
            raise ValueError(
                """Number of elements in enemy_starting_positions must be equal to num_enemies."""
            )

        self.agent = Entity(
            grid_side_length=self.grid_side_length, starting_position=agent_starting_position
        )

        self.avocados = []
        for i in range(self.num_avocados):
            avocado = Entity(
                grid_side_length=self.grid_side_length,
                starting_position=(
                    avocado_starting_positions[i] if avocado_starting_positions else None
                ),
            )
            while avocado == self.agent or any(
                existing_avocado == avocado for existing_avocado in self.avocados
            ):
                if avocado_starting_positions:
                    raise ValueError(
                        """A list of avocado starting positions was given where either: \n
                        - two or more avocados were positioned in the same cell \n
                        - at least one avocado was positioned in the same cell as the agent"""
                    )
                avocado = Entity(grid_side_length=self.grid_side_length)
            self.avocados.append(avocado)

        self.enemies = []
        for i in range(self.num_enemies):
            enemy = Entity(
                self.grid_side_length,
                starting_position=enemy_starting_positions[i] if enemy_starting_positions else None,
            )
            while enemy == self.agent:
                if enemy_starting_positions:
                    raise ValueError(
                        """A list of enemy starting positions was given where at least one enemy
                        was positioned in the same cell as the agent"""
                    )
                enemy = Entity(self.grid_side_length)
            self.enemies.append(enemy)

        self.episode_step = 0

        observation = self._get_obs()

        info = {}
        return observation, info

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.grid_side_length, self.grid_side_length, 3), dtype=np.uint8)
        for avocado in self.avocados:
            obs[avocado.y, avocado.x] = self.AVOCADO_COLOR
        obs[self.agent.y, self.agent.x] = self.AGENT_COLOR
        for enemy in self.enemies:
            obs[enemy.y, enemy.x] = self.ENEMY_COLOR

        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        self.episode_step += 1

        for enemy in self.enemies:
            if self.aggressive_enemies or self.episode_step % 3 == 0:
                enemy.move_towards_target(self.agent)
            else:
                enemy.random_action()

        self.agent.action(action)

        new_observation = self._get_obs()

        terminated = False
        truncated = False

        reward = self.STEP_PENALTY

        if any(self.agent == enemy for enemy in self.enemies):
            reward += self.ENEMY_HIT_PENALTY
            terminated = True

        elif self.agent in self.avocados:  # Checks if the agent is in a cell with an avocado
            reward += self.AVOCADO_REWARD
            self.avocados.remove(self.agent)
            terminated = not self.avocados  # Checks whether all avocados are cleared

        if self.episode_step >= 200:
            truncated = True

        info = {}

        return new_observation, reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode == "human":
            self._draw_entities()
        elif self.render_mode == "rgb_array":
            pass

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _draw_entities(self) -> None:
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))
        pix_square_size = self.window_size / self.grid_side_length

        # Drawing the avocado
        for avocado in self.avocados:
            pygame.draw.rect(
                canvas,
                self.AVOCADO_COLOR,
                pygame.Rect(
                    pix_square_size * avocado.x,
                    pix_square_size * avocado.y,
                    pix_square_size,
                    pix_square_size,
                ),
            )

        # Drawing the agent
        pygame.draw.rect(
            canvas,
            self.AGENT_COLOR,
            pygame.Rect(
                pix_square_size * self.agent.x,
                pix_square_size * self.agent.y,
                pix_square_size,
                pix_square_size,
            ),
        )

        # Drawing the enemies
        for enemy in self.enemies:
            pygame.draw.rect(
                canvas,
                self.ENEMY_COLOR,
                pygame.Rect(
                    pix_square_size * enemy.x,
                    pix_square_size * enemy.y,
                    pix_square_size,
                    pix_square_size,
                ),
            )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.render_fps)
