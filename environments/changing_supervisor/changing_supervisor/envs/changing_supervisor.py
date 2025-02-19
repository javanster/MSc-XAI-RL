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


class ChangingSupervisor(Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: Optional[LiteralString] = None,
        render_fps: int = 4,
        render_raw_pixels=False,
    ) -> None:
        super().__init__()
        self.name = "changing_supervisor"
        self.render_raw_pixels = render_raw_pixels
        self.SPRITE_MODULE_PATH: str = "changing_supervisor.envs.sprites"

        self.AGENT_COLOR: Tuple[int, int, int] = (78, 172, 248)
        self.LAVA_COLOR = (255, 69, 0)  # Red-Orange (Close to lava)
        self.TREASURE_COLOR = (255, 215, 0)  # Gold
        self.GEM_COLOR = (216, 191, 216)  # Light Purple (Thistle)
        self.GOAL_COLOR = (144, 238, 144)  # Light Green
        self.SUPERVISOR_COLORS: Dict[int, Tuple[int, int, int]] = {
            1: (0, 128, 0),  # Darker, washed-out Green
            2: (0, 0, 139),  # Darker, desaturated Blue
            3: (139, 0, 0),  # Darker, washed-out Red
            4: (153, 153, 0),  # Darker, muted Yellow
        }
        self.VALID_GOAL_POSITIONS: List[Tuple[int, int]] = [(7, 0), (14, 7), (7, 14), (0, 7)]

        self.STEP_PENALTY = -0.00125
        self.LAVA_PENALTY = -0.5
        self.TREASURE_REWARD = 0.14
        self.GEM_REWARD = 0.08
        self.GOAL_REWARD = 0.1025

        self.SUPERVISOR_DEPENDENT_REWARDS = {
            1: {"lava_penalty": 0, "treasure_reward": 0},
            2: {"lava_penalty": 0, "treasure_reward": self.TREASURE_REWARD},
            3: {"lava_penalty": self.LAVA_PENALTY, "treasure_reward": 0},
            4: {"lava_penalty": self.LAVA_PENALTY, "treasure_reward": self.TREASURE_REWARD},
        }

        self.wall_sprite_indexes = [random.randint(1, 3) for _ in range(51)]
        self.render_fps: int = render_fps
        self.grid_side_length: int = 15
        self.action_space: Discrete = Discrete(4)
        self.action_dict: Dict[int, str] = {
            0: "up",
            1: "right",
            2: "down",
            3: "left",
        }

        self.observation_space: Box = Box(
            low=0,
            high=255,
            shape=(self.grid_side_length, self.grid_side_length, 3),
            dtype=np.uint8,
        )
        self.window_size: int = 600
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode: Optional[str] = render_mode
        self.window = None
        self.clock = None

    def _place_static_objects(self):
        self.supervisors: List[Entity] = []
        self.empty_cells: List[Tuple[int, int]] = []
        self.valid_agent_starting_pos: List[Tuple[int, int]] = []
        for y, x in itertools.product(range(self.grid_side_length), range(self.grid_side_length)):
            if STATIC_MAP[y][x] == 1:
                self.supervisors.append(
                    Entity(
                        grid_side_length=self.grid_side_length,
                        starting_position=(x, y),
                        color=self.SUPERVISOR_COLORS[self.current_supervisor],
                    )
                )
            elif STATIC_MAP[y][x] == 0:
                self.empty_cells.append((x, y))
            elif STATIC_MAP[y][x] == 2:
                self.valid_agent_starting_pos.append((x, y))

        self.goal: Entity | None = Entity(
            grid_side_length=self.grid_side_length,
            color=self.GOAL_COLOR,
            starting_position=self.current_goal_position,
        )
        for supervisor in self.supervisors[:]:
            if self.goal and supervisor == self.goal:
                self.supervisors.remove(supervisor)

    def _get_pos_in_empty_space(self):
        pos = random.choice(self.empty_cells)
        self.empty_cells.remove(pos)
        return pos

    def _place_lava(self):
        self.lava: List[Entity] = []
        placed_positions = []

        while len(self.lava) < 10:
            lava_pos = self._get_pos_in_empty_space()

            # Check if the new lava position is at least 2 spaces apart from all others
            if all(
                abs(lava_pos[0] - existing_pos[0]) >= 2 or abs(lava_pos[1] - existing_pos[1]) >= 2
                for existing_pos in placed_positions
            ):
                self.lava.append(
                    Entity(
                        color=self.LAVA_COLOR,
                        grid_side_length=self.grid_side_length,
                        starting_position=lava_pos,
                    )
                )
                placed_positions.append(lava_pos)  # Keep track of placed lava positions

    def _place_treasures(self):
        self.treasures: List[Entity] = []
        for _ in range(2):
            treasure_pos = self._get_pos_in_empty_space()
            self.treasures.append(
                Entity(
                    color=self.TREASURE_COLOR,
                    grid_side_length=self.grid_side_length,
                    starting_position=treasure_pos,
                )
            )
        if self.current_supervisor == 4:
            gem_pos = self._get_pos_in_empty_space()
            self.gem = Entity(
                color=self.GEM_COLOR,
                grid_side_length=self.grid_side_length,
                starting_position=gem_pos,
            )

    def _get_new_agent_pos(self):
        opposite_mapping = {(7, 0): (7, 13), (14, 7): (1, 7), (7, 14): (7, 1), (0, 7): (13, 7)}
        return opposite_mapping.get(self.current_goal_position, None)

    def _next_room(self, supervisor_n: int):
        if supervisor_n > 4:
            raise ValueError("There are no more supervisors than 4")

        self.agent = Entity(
            color=self.AGENT_COLOR,
            grid_side_length=self.grid_side_length,
            starting_position=self._get_new_agent_pos(),
        )

        possible_goal_pos = self.VALID_GOAL_POSITIONS.copy()
        for pos in [
            (self.agent.x + 1, self.agent.y),
            (self.agent.x - 1, self.agent.y),
            (self.agent.x, self.agent.y + 1),
            (self.agent.x, self.agent.y - 1),
        ]:
            if pos in possible_goal_pos:
                possible_goal_pos.remove(pos)
        self.current_goal_position = random.choice(possible_goal_pos)
        self._place_static_objects()
        self._place_lava()
        self._place_treasures()

    def _is_goal_open(self):
        if self.current_supervisor == 2:
            return len(self.treasures) == 0
        elif self.current_supervisor == 4:
            return self.gem is None and len(self.treasures) == 0
        return True

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[Any, Any]]:
        super().reset(seed=seed)

        self.gem = None
        self.current_supervisor = 1
        self.current_goal_position = random.choice(self.VALID_GOAL_POSITIONS)

        self._place_static_objects()
        self._place_lava()
        self._place_treasures()

        agent_starting_pos = random.choice(self.valid_agent_starting_pos)
        self.agent_going_left = True if agent_starting_pos[0] >= 7 else False
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
        for supervisor in self.supervisors:
            obs[supervisor.y, supervisor.x] = supervisor.color
        if self.goal:
            obs[self.goal.y, self.goal.x] = self.goal.color
        for treasure in self.treasures:
            obs[treasure.y, treasure.x] = treasure.color
        if self.current_supervisor == 4 and self.gem:
            obs[self.gem.y, self.gem.x] = self.gem.color
        if self.current_supervisor < 3:
            for lava in self.lava:
                obs[lava.y, lava.x] = lava.color
            obs[self.agent.y, self.agent.x] = self.agent.color
        else:
            obs[self.agent.y, self.agent.x] = self.agent.color
            for lava in self.lava:
                obs[lava.y, lava.x] = lava.color

        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        self.episode_step += 1

        if action == 1:
            self.agent_going_left = False
        elif action == 3:
            self.agent_going_left = True

        original_agent_x = self.agent.x
        original_agent_y = self.agent.y
        self.agent.action(action)
        for supervisor in self.supervisors:
            if self.agent == supervisor:
                self.agent.x = original_agent_x
                self.agent.y = original_agent_y

        new_observation = self._get_obs()

        reward = self.STEP_PENALTY
        terminated = False
        truncated = False

        for lava_spot in self.lava:
            if self.agent == lava_spot:
                lava_penalty = self.SUPERVISOR_DEPENDENT_REWARDS[self.current_supervisor][
                    "lava_penalty"
                ]
                reward += lava_penalty
                if lava_penalty < 0:
                    terminated = True
                break

        for treasure in self.treasures[:]:
            if self.agent == treasure:
                self.treasures.remove(treasure)
                reward += self.SUPERVISOR_DEPENDENT_REWARDS[self.current_supervisor][
                    "treasure_reward"
                ]
                break

        if self.current_supervisor == 4 and self.agent == self.gem:
            reward += self.GEM_REWARD
            self.gem = None

        if self.agent == self.goal and self._is_goal_open():
            reward += self.GOAL_REWARD
            if self.current_supervisor == 4:
                terminated = True
            else:
                self.current_supervisor = self.current_supervisor + 1
                self._next_room(self.current_supervisor)

        if self.episode_step >= 400:
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

        canvas: pygame.Surface = pygame.Surface((self.window_size, self.window_size))
        pix_square_size: float = self.window_size / self.grid_side_length

        canvas: pygame.Surface = pygame.Surface((self.window_size, self.window_size))
        pix_square_size: float = self.window_size / self.grid_side_length

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

    def _draw_agent(self, sprites, canvas, pix_square_size):
        self._scale_and_blit_sprite(
            sprite=sprites[f"agent"],
            canvas=canvas,
            pix_square_size=pix_square_size,
            x=self.agent.x,
            y=self.agent.y,
            flip_horizontally=self.agent_going_left,
        )

    def _draw_lava(self, sprites, canvas, pix_square_size):
        for lava_spot in self.lava:
            self._scale_and_blit_sprite(
                sprite=sprites["lava"],
                canvas=canvas,
                pix_square_size=pix_square_size,
                x=lava_spot.x,
                y=lava_spot.y,
                flip_horizontally=(random.randint(0, 1) == 1),
            )

    def _draw_sprites(self, canvas: pygame.Surface, pix_square_size: float) -> None:
        sprites: Dict[str, pygame.Surface] = {
            path.name.rsplit(".", 1)[0]: self._load_sprite(path.name.rsplit(".", 1)[0])
            for path in resources.files(self.SPRITE_MODULE_PATH).iterdir()
            if path.name.endswith(".png")
        }

        canvas.fill((77, 77, 77))

        for treasure in self.treasures:
            self._scale_and_blit_sprite(
                sprite=sprites["gold"],
                canvas=canvas,
                pix_square_size=pix_square_size,
                x=treasure.x,
                y=treasure.y,
            )

        if self.gem:
            self._scale_and_blit_sprite(
                sprite=sprites["amethyst"],
                canvas=canvas,
                pix_square_size=pix_square_size,
                x=self.gem.x,
                y=self.gem.y,
            )

        if self.current_supervisor < 3:
            self._draw_lava(canvas=canvas, sprites=sprites, pix_square_size=pix_square_size)
            self._draw_agent(canvas=canvas, sprites=sprites, pix_square_size=pix_square_size)
        else:
            self._draw_agent(canvas=canvas, sprites=sprites, pix_square_size=pix_square_size)
            self._draw_lava(canvas=canvas, sprites=sprites, pix_square_size=pix_square_size)

        corner_coords = [(0, 0), (0, 14), (14, 0), (14, 14)]
        corner_supervisors = [
            supervisor
            for supervisor in self.supervisors
            if (supervisor.x, supervisor.y) in corner_coords
        ]

        edge_supervisors = [
            supervisor for supervisor in self.supervisors if supervisor not in corner_supervisors
        ]

        for i, supervisor in enumerate(edge_supervisors):
            wall_index = self.wall_sprite_indexes[i]
            rotation_degrees = 0
            if supervisor.x == 0:
                rotation_degrees = 270
            elif supervisor.y == 0:
                rotation_degrees = 180
            elif supervisor.x == 14:
                rotation_degrees = 90
            self._scale_and_blit_sprite(
                sprite=sprites[f"wall_{wall_index}_sup_{self.current_supervisor}"],
                canvas=canvas,
                pix_square_size=pix_square_size,
                x=supervisor.x,
                y=supervisor.y,
                rotation_degrees=rotation_degrees,
            )

        for supervisor in corner_supervisors:
            rotation_degrees = 0
            if supervisor.x == 14 and supervisor.y == 0:
                rotation_degrees = 180
            elif supervisor.x == 14 and supervisor.y == 14:
                rotation_degrees = 90
            elif supervisor.x == 0 and supervisor.y == 0:
                rotation_degrees = 270
            self._scale_and_blit_sprite(
                sprite=sprites[f"corner_sup_{self.current_supervisor}"],
                canvas=canvas,
                pix_square_size=pix_square_size,
                x=supervisor.x,
                y=supervisor.y,
                rotation_degrees=rotation_degrees,
            )

        if self.goal:
            rotation_degrees = 0
            if self.goal.x == 0:
                rotation_degrees = 90
            elif self.goal.y == 14:
                rotation_degrees = 180
            elif self.goal.x == 14:
                rotation_degrees = 270

            self._scale_and_blit_sprite(
                sprite=sprites[f"exit"],
                canvas=canvas,
                pix_square_size=pix_square_size,
                x=self.goal.x,
                y=self.goal.y,
                rotation_degrees=rotation_degrees,
            )

    def _scale_and_blit_sprite(
        self,
        sprite: pygame.Surface,
        canvas: pygame.Surface,
        pix_square_size: float,
        x: int,
        y: int,
        flip_horizontally: bool = False,
        rotation_degrees: int = 0,
    ) -> None:
        scaled_sprite: pygame.Surface = pygame.transform.scale(
            sprite,
            (int(pix_square_size), int(pix_square_size)),
        )

        if flip_horizontally:
            scaled_sprite = pygame.transform.flip(scaled_sprite, True, False)

        scaled_sprite = pygame.transform.rotate(
            scaled_sprite,
            rotation_degrees,
        )

        canvas.blit(
            scaled_sprite,
            (pix_square_size * x, pix_square_size * y),
        )

    def _load_sprite(self, sprite_name: str) -> pygame.Surface:
        path: Traversable = resources.files(self.SPRITE_MODULE_PATH) / f"{sprite_name}.png"
        return pygame.image.load(str(path)).convert_alpha()
