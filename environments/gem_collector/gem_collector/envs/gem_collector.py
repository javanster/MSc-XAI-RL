import itertools
import random
from importlib import resources
from importlib.abc import Traversable
from typing import Any, Callable, Dict, List, LiteralString, Optional, Tuple

import numpy as np
import pygame
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

from .entity import Entity
from .miner import Miner


class GemCollector(Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: Optional[LiteralString] = None,
        render_fps: int = 4,
        show_raw_pixels=False,
    ) -> None:
        super().__init__()
        # ----- METADATA -----
        self.name = "gem_collector"
        self.reward_range = (-1, 1)
        self.grid_side_length = 20
        self.action_space = Discrete(3)
        self.action_dict = {
            0: "left",
            1: "right",
            2: "do_nothing",
        }
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.grid_side_length, self.grid_side_length, 3),
            dtype=np.uint8,
        )
        self.SPRITE_MODULE_PATH = "gem_collector.envs.sprites"

        # ----- ENTITY COLORS -----
        self.entity_colors = {
            "agent": (20, 20, 255),
            "npc_1": (255, 255, 0),
            "npc_2": (255, 165, 0),
            "aquamarine": (127, 255, 212),
            "amethyst": (153, 102, 204),
            "emerald": (80, 200, 120),
            "rock": (158, 91, 64),
            "lava": (255, 55, 0),
            "wall_1": (77, 77, 77),
            "wall_2": (169, 169, 169),
            "wall_3": (105, 105, 105),
        }

        # ----- RENDERING VARS -----
        self.show_raw_pixels = show_raw_pixels
        self.window_size = 900
        self.render_fps = render_fps
        if render_mode and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Argument for 'render_mode' must either be 'None' or one of the following strings: {[rm for rm in self.metadata['render_modes']]}"
            )
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # ----- REWARDS -----
        self.rewards = {
            "aquamarine": 0.045,
            "amethyst": 0.033,
            "emerald": 0.022,
            "rock": -0.0004,
            "lava": -0.996,
        }

    def _reset_and_get_obj_drop_x_coordinates_for_npc_1(self) -> Dict[int, str]:

        amethyst_x_coordinates = random.randint(1, self.grid_side_length - 2)
        while amethyst_x_coordinates in self.npc_2.obj_drop_x_coordinates.keys():
            amethyst_x_coordinates = random.randint(1, self.grid_side_length - 2)
        return {amethyst_x_coordinates: "amethyst"}

    def _reset_and_get_obj_drop_x_coordinates_for_npc_2(self) -> Dict[int, str]:
        emerald_rock_x_coordinates = random.sample(range(1, self.grid_side_length - 1), 2)
        if random.random() > 0.5:
            lava_coordinate = random.randint(0, 1)
        else:
            lava_coordinate = random.randint(1, self.grid_side_length - 2)
            while (
                lava_coordinate in emerald_rock_x_coordinates
                or (lava_coordinate == 10 and not self.npc_2.is_moving_left)
                or (lava_coordinate == 9 and self.npc_2.is_moving_left)
            ):
                lava_coordinate = random.randint(1, self.grid_side_length - 2)
        return {
            emerald_rock_x_coordinates[0]: "emerald",
            emerald_rock_x_coordinates[1]: "rock",
            lava_coordinate: "lava",
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[Any, Any]]:
        super().reset(seed=seed)

        # ----- RESETTING ENTITIES -----
        self.agent = Miner(
            grid_side_length=self.grid_side_length,
            starting_position=(0, self.grid_side_length - 2),
            color=self.entity_colors["agent"],
            is_moving_left=False,
        )
        self.npc_2 = Miner(
            grid_side_length=self.grid_side_length,
            starting_position=(1, 1),
            color=self.entity_colors["npc_2"],
            is_moving_left=False,
        )
        self.npc_1 = Miner(
            grid_side_length=self.grid_side_length,
            starting_position=(self.grid_side_length - 1, 1),
            color=self.entity_colors["npc_1"],
            is_moving_left=True,
        )
        self.npc_2.obj_drop_x_coordinates = self._reset_and_get_obj_drop_x_coordinates_for_npc_2()
        self.npc_1.obj_drop_x_coordinates = self._reset_and_get_obj_drop_x_coordinates_for_npc_1()

        self.obj_lists = {"aquamarine": [], "amethyst": [], "emerald": [], "rock": [], "lava": []}

        # ----- RENDERING VARS -----
        self.miner_floor_sprite_indexes = [
            random.randint(0, 2) for _ in range(self.grid_side_length)
        ]
        self.floor_sprite_indexes = [random.randint(0, 2) for _ in range(self.grid_side_length)]
        self.active_agent_sprite = 0

        self.episode_step = 0
        observation = self._get_obs()
        info = {}
        return observation, info

    def _get_obs(self) -> np.ndarray:
        # ----- BACKGROUND PIXELS -----
        obs = np.full(
            (self.grid_side_length, self.grid_side_length, 3),
            self.entity_colors["wall_1"],
            dtype=np.uint8,
        )
        for y in range(3):
            for x in range(self.grid_side_length):
                obs[y, x] = self.entity_colors["wall_2"] if y < 2 else self.entity_colors["wall_3"]
        for x in range(self.grid_side_length):
            obs[self.grid_side_length - 1, x] = self.entity_colors["wall_3"]

        # ----- GEM AND LAVA PIXELS -----
        for obj_name in self.obj_lists.keys():
            if obj_name != "lava":
                gem_list = self.obj_lists[obj_name]
                for gem in gem_list:
                    obs[gem.y, gem.x] = gem.color

        # ----- AGENT PIXELS -----
        obs[self.agent.y, self.agent.x] = self.agent.color

        # ----- LAVA PIXELS -----
        for lava in self.obj_lists["lava"]:
            obs[lava.y, lava.x] = lava.color

        # ----- NPC PIXELS -----
        obs[self.npc_1.y, self.npc_1.x] = self.npc_1.color
        obs[self.npc_2.y, self.npc_2.x] = self.npc_2.color

        return obs

    def _is_entity_next_to_wall_x(self, entity: Entity) -> bool:
        return entity.x == 0 or entity.x == self.grid_side_length - 1

    def _drop_obj(
        self, entity_list: List[Entity], x_cord: int, entity_color: Tuple[int, int, int]
    ) -> None:
        entity_list.append(
            Entity(
                grid_side_length=self.grid_side_length,
                starting_position=(x_cord, 0),
                color=entity_color,
            )
        )

    def _gravitate_objects(self) -> None:
        for entity_list in self.obj_lists.values():
            for entity in entity_list[:]:
                if entity.y == self.grid_side_length - 1:  # Entity is on the ground
                    entity_list.remove(entity)
                else:
                    entity.action(3)  # Moves the entity 1 cell down

    def _npc_1_drop_objects(self) -> None:
        if self._is_entity_next_to_wall_x(entity=self.npc_1):
            self._drop_obj(
                entity_list=self.obj_lists["aquamarine"],
                x_cord=self.npc_1.x,
                entity_color=self.entity_colors["aquamarine"],
            )
        elif self.npc_1.x in self.npc_1.obj_drop_x_coordinates.keys():
            gem_to_drop_name = self.npc_1.obj_drop_x_coordinates[self.npc_1.x]
            gem_list = self.obj_lists[gem_to_drop_name]
            gem_color = self.entity_colors[gem_to_drop_name]
            self._drop_obj(
                entity_list=gem_list,
                x_cord=self.npc_1.x,
                entity_color=gem_color,
            )

    def _npc_2_drop_objects(self) -> None:
        if self.npc_2.x in self.npc_2.obj_drop_x_coordinates.keys():
            obj_to_drop_name = self.npc_2.obj_drop_x_coordinates[self.npc_2.x]
            obj_list = self.obj_lists[obj_to_drop_name]
            obj_color = self.entity_colors[obj_to_drop_name]
            self._drop_obj(entity_list=obj_list, x_cord=self.npc_2.x, entity_color=obj_color)

    def _move_npc(self, npc: Miner, obj_x_coord_reset: Callable[[], Dict[int, str]]) -> None:
        action = 0 if npc.is_moving_left else 1
        npc.action(action)
        if self._is_entity_next_to_wall_x(entity=npc):
            npc.is_moving_left = not npc.is_moving_left
            npc.obj_drop_x_coordinates = obj_x_coord_reset()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        self.active_agent_sprite = 0
        self.agent.action(action)
        if action == 0:
            self.agent.is_moving_left = True
        elif action == 1:
            self.agent.is_moving_left = False

        self._gravitate_objects()
        self._npc_1_drop_objects()
        self._npc_2_drop_objects()
        self._move_npc(
            npc=self.npc_2, obj_x_coord_reset=self._reset_and_get_obj_drop_x_coordinates_for_npc_2
        )
        self._move_npc(
            npc=self.npc_1,
            obj_x_coord_reset=self._reset_and_get_obj_drop_x_coordinates_for_npc_1,
        )

        reward = 0
        terminated = False
        truncated = False

        for obj_type in self.obj_lists.keys():
            obj_list = self.obj_lists[obj_type]
            for obj in obj_list[:]:
                if obj == self.agent:
                    obj_list.remove(obj)
                    reward += self.rewards[obj_type]
                    if obj_type == "rock":
                        self.active_agent_sprite = 1
                    elif obj_type == "lava":
                        terminated = True
                        break
                    else:
                        self.active_agent_sprite = 2

        if self.episode_step >= 190:  # max 10 complete rounds back and forth
            truncated = True
            terminated = True

        info = {}
        self.episode_step += 1
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

    def _load_sprite(self, sprite_name: str) -> pygame.Surface:
        path = resources.files(self.SPRITE_MODULE_PATH) / f"{sprite_name}.png"
        return pygame.image.load(str(path)).convert_alpha()

    def _scale_and_blit_sprite(
        self,
        sprite,
        canvas,
        pix_square_size: float,
        x: int,
        y: int,
        random_rotate: bool = False,
        flip_horizontally: bool = False,
    ) -> None:
        sprite_roatation_degrees = [d for d in range(0, 316, 45)]
        scaled_sprite = pygame.transform.scale(
            sprite,
            (int(pix_square_size), int(pix_square_size)),
        )

        if random_rotate:
            scaled_sprite = pygame.transform.rotate(
                scaled_sprite, random.choice(sprite_roatation_degrees)
            )

        if flip_horizontally:
            scaled_sprite = pygame.transform.flip(scaled_sprite, True, False)

        canvas.blit(
            scaled_sprite,
            (pix_square_size * x, pix_square_size * y),
        )

    def _draw_sprites(self, canvas, pix_square_size) -> None:
        sprites = {
            path.name.rsplit(".", 1)[0]: self._load_sprite(path.name.rsplit(".", 1)[0])
            for path in resources.files(self.SPRITE_MODULE_PATH).iterdir()
            if path.name.endswith(".png")
        }

        canvas.fill(self.entity_colors["wall_1"])

        for x, y in itertools.product([n for n in range(self.grid_side_length)], [0, 1, 2]):
            pygame.draw.rect(
                canvas,
                self.entity_colors["wall_2"],
                pygame.Rect(
                    pix_square_size * x,
                    pix_square_size * y,
                    pix_square_size,
                    pix_square_size,
                ),
            )

        for x, y in itertools.product(range(self.grid_side_length), [2, self.grid_side_length - 1]):
            # Randomly pick a floor sprite
            floor_sprite = [sprites["wall_1"], sprites["wall_2"], sprites["wall_3"]][
                self.miner_floor_sprite_indexes[x]
            ]
            self._scale_and_blit_sprite(
                sprite=floor_sprite, canvas=canvas, pix_square_size=pix_square_size, x=x, y=y
            )

        for obj_type in self.obj_lists.keys():
            if obj_type == "lava":
                break
            gem_list = self.obj_lists[obj_type]
            for gem in gem_list:
                self._scale_and_blit_sprite(
                    sprite=sprites[obj_type],
                    canvas=canvas,
                    pix_square_size=pix_square_size,
                    x=gem.x,
                    y=gem.y,
                    random_rotate=True,
                )

        for rock in self.obj_lists["rock"]:
            self._scale_and_blit_sprite(
                sprite=sprites["rock"],
                canvas=canvas,
                pix_square_size=pix_square_size,
                x=rock.x,
                y=rock.y,
                random_rotate=True,
            )

        self._scale_and_blit_sprite(
            sprite=sprites[f"agent_{self.active_agent_sprite}"],
            canvas=canvas,
            pix_square_size=pix_square_size,
            x=self.agent.x,
            y=self.agent.y,
            flip_horizontally=self.agent.is_moving_left,
        )

        for lava in self.obj_lists["lava"]:

            self._scale_and_blit_sprite(
                sprite=sprites["lava"],
                canvas=canvas,
                pix_square_size=pix_square_size,
                x=lava.x,
                y=lava.y,
                flip_horizontally=self.episode_step % 2 == 1,
            )

        self._scale_and_blit_sprite(
            sprite=sprites["npc_1"],
            canvas=canvas,
            pix_square_size=pix_square_size,
            x=self.npc_1.x,
            y=self.npc_1.y,
            flip_horizontally=self.npc_1.is_moving_left,
        )

        self._scale_and_blit_sprite(
            sprite=sprites["npc_2"],
            canvas=canvas,
            pix_square_size=pix_square_size,
            x=self.npc_2.x,
            y=self.npc_2.y,
            flip_horizontally=self.npc_2.is_moving_left,
        )

    def _draw_entities(self) -> None:
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        pix_square_size = self.window_size / self.grid_side_length

        if self.show_raw_pixels:
            obs_grid = self._get_obs()

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
