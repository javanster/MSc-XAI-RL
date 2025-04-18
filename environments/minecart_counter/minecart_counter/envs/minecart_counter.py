import itertools
import random
from importlib import resources
from importlib.abc import Traversable
from typing import Any, Dict, List, LiteralString, Optional, Tuple

import numpy as np
import pygame
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

from .entity import Entity
from .static_map import STATIC_MAP


class MinecartCounter(Env):
    """
    A grid-based environment where an agent must navigate toward a goal based on the number of minecarts (orange boxes).

    There are eight goals which each terminate the episode when reached by the agent. The reward received
    is based on whether the goal was the correct one or not. Each episode, one of the goals is selected
    as the correct one at random. The chosen goal decides how many minecarts are placed in the
    environment, making the number of minecarts a direct indication of which goal is correct.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: Optional[LiteralString] = None,
        render_fps: int = 4,
        scatter_minecarts: bool = True,
        render_raw_pixels: bool = False,
    ) -> None:
        super().__init__()
        self.name = "minecart_counter"

        self.AGENT_COLOR: Tuple[int, int, int] = (78, 172, 248)
        self.WALL_COLOR: Tuple[int, int, int] = (77, 77, 77)
        self.MINECART_COLOR: Tuple[int, int, int] = (255, 180, 65)
        self.GOAL_COLORS: Dict[int, Tuple[int, int, int]] = {
            1: (165, 172, 31),  # Correct goal if 1 minecart = 1 * 31 = 31
            2: (165, 172, 62),  # Correct goal if 2 minecarts = 2 * 31 = 62
            3: (165, 172, 93),  # Correct goal if 3 minecarts = 3 * 31 = 93
            4: (165, 172, 124),  # Correct goal if 4 minecarts = 4 * 31 = 124
            5: (165, 172, 155),  # Correct goal if 5 minecarts = 5 * 31 = 155
            6: (165, 172, 186),  # Correct goal if 6 minecarts = 6 * 31 = 186
            7: (165, 172, 217),  # Correct goal if 7 minecarts = 7 * 31 = 217
            8: (165, 172, 248),  # Correct goal if 8 minecarts = 8 * 31 = 248
        }
        self.GOAL_POSITIONS: Dict[int, Tuple[int, int]] = {
            1: (7, 0),
            2: (13, 1),
            3: (14, 7),
            4: (13, 13),
            5: (7, 14),
            6: (1, 13),
            7: (0, 7),
            8: (1, 1),
        }

        self.ENTITY_COLOR_MAP = {
            "agent": self.AGENT_COLOR,
            "wall": self.WALL_COLOR,
            "minecart": self.MINECART_COLOR,
            **{f"goal_{i}": color for i, color in self.GOAL_COLORS.items()},
        }

        self.SCATTER_MINECARTS: bool = scatter_minecarts
        self.render_raw_pixels = render_raw_pixels
        self.SPRITE_MODULE_PATH: str = "minecart_counter.envs.sprites"

        self.render_fps: int = render_fps
        self.grid_side_length: int = 15
        self.action_space: Discrete = Discrete(8)
        self.action_dict: Dict[int, str] = {
            0: "up",
            1: "up_right",
            2: "right",
            3: "down_right",
            4: "down",
            5: "down_left",
            6: "left",
            7: "up_left",
        }
        self.observation_space: Box = Box(
            low=0,
            high=255,
            shape=(self.grid_side_length, self.grid_side_length, 3),
            dtype=np.uint8,
        )
        self.window_size: int = 900
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode: Optional[str] = render_mode
        self.window = None
        self.clock = None

    def _place_static_objects(self):
        self.walls: List[Entity] = []
        self.empty_cells: List[Tuple[int, int]] = []
        for y, x in itertools.product(range(self.grid_side_length), range(self.grid_side_length)):
            if STATIC_MAP[y][x] == 1:
                self.walls.append(
                    Entity(
                        grid_side_length=self.grid_side_length,
                        starting_position=(x, y),
                        color=self.WALL_COLOR,
                    )
                )
            elif STATIC_MAP[y][x] == 0:
                self.empty_cells.append((x, y))

        self.goals: List[Entity] = []
        for i in range(8):
            self.goals.append(
                Entity(
                    grid_side_length=self.grid_side_length,
                    color=self.GOAL_COLORS[i + 1],
                    starting_position=self.GOAL_POSITIONS[i + 1],
                )
            )

    def _place_minecarts_scattered(self, minecarts_n: int) -> None:
        for _ in range(minecarts_n):
            minecart_pos = random.choice(self.empty_cells)
            self.minecarts.append(
                Entity(
                    grid_side_length=self.grid_side_length,
                    color=self.MINECART_COLOR,
                    starting_position=(minecart_pos),
                )
            )
            self.empty_cells.remove(minecart_pos)

    def _place_minecarts_clustered(self, minecarts_n: int) -> None:
        first_minecart_pos: Tuple[int, int] = random.choice(self.empty_cells)
        self.minecarts.append(
            Entity(
                grid_side_length=self.grid_side_length,
                color=self.MINECART_COLOR,
                starting_position=(first_minecart_pos),
            )
        )
        self.empty_cells.remove(first_minecart_pos)

        while len(self.minecarts) < minecarts_n:
            adjacent_positions = set()
            for minecart in self.minecarts:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, down, left, right
                    nx, ny = minecart.x + dx, minecart.y + dy
                    if (nx, ny) in self.empty_cells:
                        adjacent_positions.add((nx, ny))

            # If no valid adjacent position exists, stop placement
            if not adjacent_positions:
                break

            # Choose a random position from the available adjacent positions
            next_minecart_pos = random.choice(list(adjacent_positions))
            self.minecarts.append(
                Entity(
                    grid_side_length=self.grid_side_length,
                    color=self.MINECART_COLOR,
                    starting_position=(next_minecart_pos),
                )
            )
            self.empty_cells.remove(next_minecart_pos)

    def _place_minecarts(self) -> None:
        self.minecarts: List[Entity] = []
        minecarts_n: int = self.target_direction

        if self.SCATTER_MINECARTS:
            self._place_minecarts_scattered(minecarts_n=minecarts_n)
        else:
            self._place_minecarts_clustered(minecarts_n=minecarts_n)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[Any, Any]]:
        super().reset(seed=seed)

        self.target_direction = random.randint(1, 8)
        self.minecart_horizontal_flips = [
            random.randint(0, 1) == 1 for _ in range(1_000_000)
        ]  # Not great, not terrible.

        self._place_static_objects()
        self._place_minecarts()
        self.agent_going_left = True

        agent_starting_pos = (7, 7)
        self.agent = Entity(
            grid_side_length=self.grid_side_length,
            starting_position=(agent_starting_pos),
            color=self.AGENT_COLOR,
        )
        self.episode_step = 0

        observation = self._get_obs()

        info = {}
        return observation, info

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.grid_side_length, self.grid_side_length, 3), dtype=np.uint8)
        for wall in self.walls:
            obs[wall.y, wall.x] = self.WALL_COLOR
        for goal in self.goals:
            obs[goal.y, goal.x] = goal.color
        for minecart in self.minecarts:
            obs[minecart.y, minecart.x] = minecart.color
        obs[self.agent.y, self.agent.x] = self.AGENT_COLOR

        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        self.episode_step += 1

        if action in [1, 2, 3]:
            self.agent_going_left = False
        elif action in [5, 6, 7]:
            self.agent_going_left = True

        original_agent_x = self.agent.x
        original_agent_y = self.agent.y
        self.agent.action(action)
        for wall in self.walls:
            if self.agent == wall:
                self.agent.x = original_agent_x
                self.agent.y = original_agent_y

        for minecart in self.minecarts:
            if self.agent == minecart:
                self.agent.x = original_agent_x
                self.agent.y = original_agent_y

        reward = -0.005
        terminated = False
        truncated = False
        goal_reached = False
        correct_goal_reached = False

        goal_reward = 0
        for i, goal in enumerate(self.goals):
            if self.agent == goal:
                goal_reached = True
                goal_reward = 0.21
                if i + 1 == self.target_direction:
                    correct_goal_reached = True
                    goal_reward = 1.01
                terminated = True
                break
        reward += goal_reward

        if self.episode_step >= 200:
            truncated = True

        info = {
            "goal_reached": goal_reached,
            "correct_goal_reached": correct_goal_reached,
            "truncated": truncated,
        }

        new_observation = self._get_obs()

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
        pix_square_size = self.window_size / self.grid_side_length

        if self.render_raw_pixels:
            obs_grid: np.ndarray = self._get_obs()

            for y, x in itertools.product(range(self.grid_side_length), repeat=2):
                color = tuple(obs_grid[y, x])
                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(
                        x * pix_square_size,
                        y * pix_square_size,
                        pix_square_size,
                        pix_square_size,
                    ),
                )

        else:
            self._draw_sprites(canvas=canvas, pix_square_size=pix_square_size)

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.render_fps)

    def _draw_sprites(self, canvas: pygame.Surface, pix_square_size: float) -> None:
        sprites: Dict[str, pygame.Surface] = {
            path.name.rsplit(".", 1)[0]: self._load_sprite(path.name.rsplit(".", 1)[0])
            for path in resources.files(self.SPRITE_MODULE_PATH).iterdir()
            if path.name.endswith(".png")
        }

        canvas.fill((22, 22, 22))

        # Drawing the walls
        for wall in self.walls:
            pygame.draw.rect(
                canvas,
                wall.color,
                pygame.Rect(
                    pix_square_size * wall.x,
                    pix_square_size * wall.y,
                    pix_square_size,
                    pix_square_size,
                ),
            )

        for i, minecart in enumerate(self.minecarts):
            minecart_sprite = sprites["minecart"]
            self._scale_and_blit_sprite(
                canvas=canvas,
                sprite=minecart_sprite,
                pix_square_size=pix_square_size,
                flip_horizontally=self.minecart_horizontal_flips[i],
                x=minecart.x,
                y=minecart.y,
            )

        for i, goal in enumerate(self.goals):
            goal_sprite = sprites[f"{i + 1}_sign"]
            self._scale_and_blit_sprite(
                canvas=canvas,
                sprite=goal_sprite,
                pix_square_size=pix_square_size,
                x=goal.x,
                y=goal.y,
            )

        agent_sprite = sprites["agent"]
        self._scale_and_blit_sprite(
            canvas=canvas,
            sprite=agent_sprite,
            pix_square_size=pix_square_size,
            x=self.agent.x,
            y=self.agent.y,
            flip_horizontally=self.agent_going_left,
        )

    def _scale_and_blit_sprite(
        self,
        sprite: pygame.Surface,
        canvas: pygame.Surface,
        pix_square_size: float,
        x: int,
        y: int,
        flip_horizontally: bool = False,
    ) -> None:
        scaled_sprite: pygame.Surface = pygame.transform.scale(
            sprite,
            (int(pix_square_size), int(pix_square_size)),
        )

        if flip_horizontally:
            scaled_sprite = pygame.transform.flip(scaled_sprite, True, False)

        canvas.blit(
            scaled_sprite,
            (pix_square_size * x, pix_square_size * y),
        )

    def _load_sprite(self, sprite_name: str) -> pygame.Surface:
        path: Traversable = resources.files(self.SPRITE_MODULE_PATH) / f"{sprite_name}.png"
        return pygame.image.load(str(path))

    def _set_entity(self, entity_type: str, pos: Tuple[int, int]):
        if entity_type == "agent":
            self.agent = Entity(
                color=self.ENTITY_COLOR_MAP[entity_type],
                grid_side_length=self.grid_side_length,
                starting_position=pos,
            )
        elif entity_type == "minecart":
            self.minecarts.append(
                Entity(
                    grid_side_length=self.grid_side_length,
                    color=self.ENTITY_COLOR_MAP[entity_type],
                    starting_position=pos,
                )
            )

    def set_state_based_on_obs_grid(self, obs_grid: np.ndarray) -> None:
        """
        Update the environment's state based on an observation grid.

        This method clears the current lists of minecarts, and then
        iterates over each cell in the provided observation grid. For each cell, if its
        color matches one of the colors in ENTITY_COLOR_MAP, the corresponding entity is
        created at that grid position using the _set_entity helper method.

        Parameters
        ----------
        obs_grid : np.ndarray
            A NumPy array of shape (grid_side_length, grid_side_length, 3) representing the
            observation grid, where each cell contains an RGB color tuple.

        Returns
        -------
        None
        """
        self.minecarts = []

        for y, x in itertools.product(range(self.grid_side_length), repeat=2):
            color = tuple(obs_grid[y, x])
            for entity_type, color_of_type in self.ENTITY_COLOR_MAP.items():
                if color == color_of_type:
                    self._set_entity(entity_type=entity_type, pos=(x, y))
