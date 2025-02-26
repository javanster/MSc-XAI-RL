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


class GoldRunMini(Env):
    """
    A Gymnasium environment where the goal of the agent is to collect gold to open passages and go
    through them.

    There are two rooms. The first room has one gold chunk in it, and two passages. The passages
    only open once the agent has collected the gold chunk. Once open, the agent may choose between
    wither of the two passages. The purple passage terminates the episode. The green passage leads
    the agent into the next room, whihc contains yet another gold chunk, a closed passage.
    Additionally, this room also contains spots of lava which the agent must avoid. The goal in
    this room remains the same - get the gold and go through the passage, terminating the episode.

    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: Optional[LiteralString] = None,
        render_fps: int = 4,
        render_raw_pixels: bool = False,
        disable_early_termination: bool = False,
        only_second_room: bool = False,
        no_lava_termination: bool = False,
        lava_spots: int = 8,
    ) -> None:
        super().__init__()
        self.name = "gold_run_mini"
        self.render_raw_pixels = render_raw_pixels
        self.SPRITE_MODULE_PATH: str = "gold_run_mini.envs.sprites"

        self.AGENT_COLOR: Tuple[int, int, int] = (78, 172, 248)
        self.LAVA_COLOR = (255, 69, 0)  # Red-Orange
        self.GOLD_COLOR = (255, 215, 0)  # Gold
        self.CLOSED_PASSAGE_COLOR = (34, 139, 34)  # Dark Green
        self.OPEN_PASSAGE_COLOR = (144, 238, 144)  # Light Green
        self.CLOSED_EARLY_TERMINATION_PASSAGE_COLOR = (75, 0, 130)  # Dark Purple
        self.OPEN_EARLY_TERMINATION_PASSAGE_COLOR = (221, 160, 221)  # Light Purple
        self.WALL_COLOR = (77, 77, 77)
        self.VALID_PASSAGE_POSITIONS: List[Tuple[int, int]] = [(5, 0), (10, 5), (5, 10), (0, 5)]

        self.STEP_PENALTY = -0.001 if no_lava_termination else -0.0026
        self.LAVA_PENALTY = -0.03 if no_lava_termination else -0.48
        self.GOLD_REWARD = 0.309 if no_lava_termination else 0.2617
        self.PASSAGE_REWARD = 0.2 if no_lava_termination else 0.2617
        self.EARLY_TERM_PASSAGE_REWARD = 0.2 if no_lava_termination else 0.2617

        if lava_spots > 8 or lava_spots < 0:
            raise ValueError("The max number of lava spots is 8, and the min is 0.")
        self.lava_spots = lava_spots
        self.no_lava_termination = no_lava_termination
        self.disable_early_termination = disable_early_termination
        self.only_second_room = only_second_room
        self.wall_sprite_indexes = [random.randint(1, 3) for _ in range(36)]
        self.render_fps: int = render_fps
        self.grid_side_length: int = 11
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
        self.walls: List[Entity] = []
        self.empty_cells: List[Tuple[int, int]] = []
        self.valid_agent_starting_pos: List[Tuple[int, int]] = []
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
            elif STATIC_MAP[y][x] == 2:
                self.valid_agent_starting_pos.append((x, y))

    def _get_pos_in_empty_space(self):
        pos = random.choice(self.empty_cells)
        self.empty_cells.remove(pos)
        return pos

    def _place_lava(self):
        self.lava: List[Entity] = []
        placed_positions = []

        while len(self.lava) < self.lava_spots:
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
                placed_positions.append(lava_pos)

    def _place_gold_chunks(self):
        self.gold_chunks: List[Entity] = []
        for _ in range(1):
            gold_pos = self._get_pos_in_empty_space()
            self.gold_chunks.append(
                Entity(
                    color=self.GOLD_COLOR,
                    grid_side_length=self.grid_side_length,
                    starting_position=gold_pos,
                )
            )

    def _get_new_agent_pos(self):
        opposite_mapping = {(5, 0): (5, 9), (10, 5): (1, 5), (5, 10): (5, 1), (0, 5): (9, 5)}
        return opposite_mapping.get(self.current_passage_position, (5, 9))

    def _next_room(self, room_n: int):
        if room_n > 2:
            raise ValueError("There are no more rooms than 2")

        agent_starting_pos = self._get_new_agent_pos()
        self.agent_going_left = True if agent_starting_pos[0] >= 5 else False

        self.agent = Entity(
            color=self.AGENT_COLOR,
            grid_side_length=self.grid_side_length,
            starting_position=agent_starting_pos,
        )

        possible_passage_pos = self.VALID_PASSAGE_POSITIONS.copy()
        non_valid_passage_pos = [
            (self.agent.x + 1, self.agent.y),
            (self.agent.x - 1, self.agent.y),
            (self.agent.x, self.agent.y + 1),
            (self.agent.x, self.agent.y - 1),
        ]
        for pos in non_valid_passage_pos:
            if pos in possible_passage_pos:
                possible_passage_pos.remove(pos)
        self.current_passage_position = random.choice(possible_passage_pos)

        possible_early_term_passage_pos = [
            pos for pos in possible_passage_pos if pos != self.current_passage_position
        ]
        self.current_early_term_passage_pos = random.choice(possible_early_term_passage_pos)

        self._place_static_objects()
        self._place_gold_chunks()

        if room_n > 1:
            self._place_lava()

    def _is_passage_open(self):
        return len(self.gold_chunks) == 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[Any, Any]]:
        super().reset(seed=seed)

        self.current_room = 2 if self.only_second_room else 1
        self.current_passage_position = random.choice(self.VALID_PASSAGE_POSITIONS)
        self.lava: List[Entity] = []
        self._next_room(self.current_room)

        self.passage: Entity | None = Entity(
            grid_side_length=self.grid_side_length,
            color=self.CLOSED_PASSAGE_COLOR,
            starting_position=self.current_passage_position,
        )

        if self.only_second_room:
            self.early_term_passage: Entity | None = None
        else:
            self.early_term_passage: Entity | None = Entity(
                color=self.CLOSED_EARLY_TERMINATION_PASSAGE_COLOR,
                grid_side_length=self.grid_side_length,
                starting_position=self.current_early_term_passage_pos,
            )

        for wall in self.walls[:]:
            if (self.passage and wall == self.passage) or (
                self.early_term_passage and wall == self.early_term_passage
            ):
                self.walls.remove(wall)

        self.episode_step = 0

        observation = self._get_obs()

        info = {}
        return observation, info

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.grid_side_length, self.grid_side_length, 3), dtype=np.uint8)
        for wall in self.walls:
            obs[wall.y, wall.x] = wall.color
        if self.passage:
            obs[self.passage.y, self.passage.x] = self.passage.color
        if self.early_term_passage:
            obs[self.early_term_passage.y, self.early_term_passage.x] = (
                self.early_term_passage.color
            )
        for gold in self.gold_chunks:
            obs[gold.y, gold.x] = gold.color
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
        for wall in self.walls:
            if self.agent == wall:
                self.agent.x = original_agent_x
                self.agent.y = original_agent_y
        if not self._is_passage_open() and (
            self.agent == self.passage or self.agent == self.early_term_passage
        ):
            self.agent.x = original_agent_x
            self.agent.y = original_agent_y

        new_observation = self._get_obs()

        reward = self.STEP_PENALTY
        terminated = False
        truncated = False

        info = {
            "lava_stepped_on": 0,
            "gold_picked_up": 0,
            "exited_in_final_passage": False,
            "went_to_next_room": False,
            "exited_in_early_termination_passage": False,
        }

        for lava_spot in self.lava:
            if self.agent == lava_spot:
                lava_penalty = self.LAVA_PENALTY
                reward += lava_penalty
                terminated = False if self.no_lava_termination else True
                info["lava_stepped_on"] += 1
                break

        for gold in self.gold_chunks[:]:
            if self.agent == gold:
                self.gold_chunks.remove(gold)
                reward += self.GOLD_REWARD
                info["gold_picked_up"] += 1
                break

        if self.agent == self.passage and self._is_passage_open():
            reward += self.PASSAGE_REWARD
            if self.current_room == 2:
                terminated = True
                info["exited_in_final_passage"] = True
            else:
                self.current_room = self.current_room + 1
                self._next_room(self.current_room)
                info["went_to_next_room"] = True

        if (
            not self.disable_early_termination
            and self.early_term_passage
            and self.agent == self.early_term_passage
        ):
            reward += self.EARLY_TERM_PASSAGE_REWARD
            terminated = True
            info["exited_in_early_termination_passage"] = True

        self.passage: Entity | None = Entity(
            grid_side_length=self.grid_side_length,
            color=self.OPEN_PASSAGE_COLOR if self._is_passage_open() else self.CLOSED_PASSAGE_COLOR,
            starting_position=self.current_passage_position,
        )

        if self.current_room < 2:
            self.early_term_passage: Entity | None = Entity(
                grid_side_length=self.grid_side_length,
                color=(
                    self.OPEN_EARLY_TERMINATION_PASSAGE_COLOR
                    if self._is_passage_open()
                    else self.CLOSED_EARLY_TERMINATION_PASSAGE_COLOR
                ),
                starting_position=self.current_early_term_passage_pos,
            )
        else:
            self.early_term_passage: Entity | None = None

        for wall in self.walls[:]:
            if (self.passage and wall == self.passage) or (
                self.early_term_passage and wall == self.early_term_passage
            ):
                self.walls.remove(wall)

        if self.episode_step >= 200:
            truncated = True

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

        canvas.fill((45, 45, 45))

        for gold in self.gold_chunks:
            self._scale_and_blit_sprite(
                sprite=sprites["gold"],
                canvas=canvas,
                pix_square_size=pix_square_size,
                x=gold.x,
                y=gold.y,
            )

        self._draw_agent(canvas=canvas, sprites=sprites, pix_square_size=pix_square_size)
        self._draw_lava(canvas=canvas, sprites=sprites, pix_square_size=pix_square_size)

        corner_coords = [(0, 0), (0, 10), (10, 0), (10, 10)]
        corner_walls = [wall for wall in self.walls if (wall.x, wall.y) in corner_coords]

        edge_walls = [wall for wall in self.walls if wall not in corner_walls]

        for i, wall in enumerate(edge_walls):
            wall_index = self.wall_sprite_indexes[i]
            rotation_degrees = 0
            if wall.x == 0:
                rotation_degrees = 270
            elif wall.y == 0:
                rotation_degrees = 180
            elif wall.x == 10:
                rotation_degrees = 90
            self._scale_and_blit_sprite(
                sprite=sprites[f"wall_{wall_index}"],
                canvas=canvas,
                pix_square_size=pix_square_size,
                x=wall.x,
                y=wall.y,
                rotation_degrees=rotation_degrees,
            )

        for wall in corner_walls:
            rotation_degrees = 0
            if wall.x == 10 and wall.y == 0:
                rotation_degrees = 180
            elif wall.x == 10 and wall.y == 10:
                rotation_degrees = 90
            elif wall.x == 0 and wall.y == 0:
                rotation_degrees = 270
            self._scale_and_blit_sprite(
                sprite=sprites[f"corner"],
                canvas=canvas,
                pix_square_size=pix_square_size,
                x=wall.x,
                y=wall.y,
                rotation_degrees=rotation_degrees,
            )

        if self.passage:
            rotation_degrees = 0
            if self.passage.x == 0:
                rotation_degrees = 90
            elif self.passage.y == 10:
                rotation_degrees = 180
            elif self.passage.x == 10:
                rotation_degrees = 270

            self._scale_and_blit_sprite(
                sprite=(
                    sprites["passage_open"]
                    if self._is_passage_open()
                    else sprites["passage_closed"]
                ),
                canvas=canvas,
                pix_square_size=pix_square_size,
                x=self.passage.x,
                y=self.passage.y,
                rotation_degrees=rotation_degrees,
            )

        if self.early_term_passage:
            rotation_degrees = 0
            if self.early_term_passage.x == 0:
                rotation_degrees = 90
            elif self.early_term_passage.y == 10:
                rotation_degrees = 180
            elif self.early_term_passage.x == 10:
                rotation_degrees = 270

            self._scale_and_blit_sprite(
                sprite=(
                    sprites["early_term_passage_open"]
                    if self._is_passage_open()
                    else sprites["early_term_passage_closed"]
                ),
                canvas=canvas,
                pix_square_size=pix_square_size,
                x=self.early_term_passage.x,
                y=self.early_term_passage.y,
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
