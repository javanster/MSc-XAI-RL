import random
from typing import Any, Dict, LiteralString, Optional, Tuple

import numpy as np
import pygame
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

from .entity import Entity


class FruitRun(Env):
    """
    A custom Gymnasium environment where the goal of the agent is to avoid a moving enemy and eat
    the apple or the mango (if present).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: Optional[LiteralString] = None,
        render_fps: int = 4,
        agent_starting_position: Optional[Tuple[int, int]] = None,
        mango_always_present: bool = False,
    ) -> None:
        super().__init__()
        self.name = "fruit_run"
        self.STEP_PENALTY = -0.0035
        self.ENEMY_HIT_PENALTY = -0.3
        self.APPLE_REWARD = 0.5
        self.MANGO_REWARD = 1.0035
        self.reward_range = (-1, 1)

        self.mango_always_present = mango_always_present

        self.APPLE_COLOR = (0, 255, 0)
        self.MANGO_COLOR = (255, 215, 0)
        self.AGENT_COLOR = (78, 172, 248)
        self.ENEMY_COLOR = (255, 0, 0)

        self.render_fps = render_fps
        self.agent_starting_position = agent_starting_position
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
    ):
        super().reset(seed=seed)

        self.agent = Entity(
            grid_side_length=self.grid_side_length,
            starting_position=(
                self.agent_starting_position if self.agent_starting_position else None
            ),
        )

        self.apple = Entity(
            grid_side_length=self.grid_side_length,
        )
        while self.apple == self.agent:
            self.apple = Entity(
                grid_side_length=self.grid_side_length,
            )

        self.mango = None
        if self.mango_always_present or random.random() >= 0.5:
            self.mango = Entity(
                grid_side_length=self.grid_side_length,
            )
            while self.mango == self.agent or self.mango == self.apple:
                self.mango = Entity(
                    grid_side_length=self.grid_side_length,
                )

        self.num_enemies = 1

        self.enemies = []
        for _ in range(self.num_enemies):
            enemy = Entity(self.grid_side_length)
            while enemy == self.agent:
                enemy = Entity(self.grid_side_length)
            self.enemies.append(enemy)

        self.episode_step = 0

        observation = self._get_obs()

        info = {}
        return observation, info

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.grid_side_length, self.grid_side_length, 3), dtype=np.uint8)
        obs[self.apple.y, self.apple.x] = self.APPLE_COLOR
        if self.mango:
            obs[self.mango.y, self.mango.x] = self.MANGO_COLOR
        obs[self.agent.y, self.agent.x] = self.AGENT_COLOR
        for enemy in self.enemies:
            obs[enemy.y, enemy.x] = self.ENEMY_COLOR

        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        self.episode_step += 1

        for enemy in self.enemies:
            if self.episode_step % 3 == 0:
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

        elif self.agent == self.apple:
            reward += self.APPLE_REWARD
            terminated = True

        elif self.mango and self.agent == self.mango:
            reward += self.MANGO_REWARD
            terminated = True

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

        # Drawing the apple
        pygame.draw.rect(
            canvas,
            self.APPLE_COLOR,
            pygame.Rect(
                pix_square_size * self.apple.x,
                pix_square_size * self.apple.y,
                pix_square_size,
                pix_square_size,
            ),
        )

        # Drawing the mango
        if self.mango:
            pygame.draw.rect(
                canvas,
                self.MANGO_COLOR,
                pygame.Rect(
                    pix_square_size * self.mango.x,
                    pix_square_size * self.mango.y,
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
