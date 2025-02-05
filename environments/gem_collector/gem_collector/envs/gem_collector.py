import itertools
import random
from importlib import resources
from typing import Any, Dict, List, LiteralString, Optional, Tuple

import numpy as np
import pygame
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

from .entity import Entity


class GemCollector(Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: Optional[LiteralString] = None,
        render_fps: int = 4,
        show_raw_pixels=False,
    ) -> None:
        super().__init__()
        self.name = "gem_collector"
        self.reward_range = (-1, 1)
        self.show_raw_pixels = show_raw_pixels

        self.AGENT_COLOR = (20, 20, 255)
        self.NPC_1_COLOR = (255, 255, 0)  # Yellow
        self.NPC_2_COLOR = (255, 165, 0)  # Orange
        self.AQUAMARINE_COLOR = (127, 255, 212)
        self.AMETHYST_COLOR = (153, 102, 204)
        self.EMERALD_COLOR = (80, 200, 120)
        self.ROCK_COLOR = (158, 91, 64)
        self.GROUND_COLOR_1 = (169, 169, 169)  # Light gray
        self.GROUND_COLOR_2 = (105, 105, 105)  # Dark gray
        self.LAVA_COLOR = (255, 55, 0)
        self.BACKGROUND_COLOR = (77, 77, 77)

        self.render_fps = render_fps
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
        self.window_size = 900
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.rewards = {
            "aquamarine": 0.045,
            "amethyst": 0.033,
            "emerald": 0.022,
            "rock": -0.0004,
            "lava": -0.996,
        }

    def _set_gem_drop_x_coordinates_for_npc_1(self):

        amethyst_x_coordinates = random.randint(1, self.grid_side_length - 2)
        while amethyst_x_coordinates in self.npc_2_gem_drop_coordinates.keys():
            amethyst_x_coordinates = random.randint(1, self.grid_side_length - 2)

        return {amethyst_x_coordinates: "amethyst"}

    def _set_gem_drop_x_coordinates_for_npc_2(self):
        emerald_rock_x_coordinates = random.sample(range(1, self.grid_side_length - 1), 2)

        return {emerald_rock_x_coordinates[0]: "emerald", emerald_rock_x_coordinates[1]: "rock"}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[Any, Any]]:
        super().reset(seed=seed)

        self.gems_dropped = {"aquamarine": 0, "amethyst": 0, "emerald": 0, "rock": 0}
        self.gems_collected = {"aquamarine": 0, "amethyst": 0, "emerald": 0, "rock": 0}

        self.agent = Entity(
            grid_side_length=self.grid_side_length,
            starting_position=(0, self.grid_side_length - 2),
        )

        self.npc_1 = Entity(
            grid_side_length=self.grid_side_length,
            starting_position=(self.grid_side_length - 1, 1),
        )
        self.npc_1_moving_left = True

        self.npc_2 = Entity(
            grid_side_length=self.grid_side_length,
            starting_position=(1, 1),
        )
        self.npc_2_moving_left = False
        self.agent_moving_left = False
        self.agent_collected = "none"

        self.npc_2_gem_drop_coordinates = self._set_gem_drop_x_coordinates_for_npc_2()

        self.npc_1_gem_drop_coordinates = self._set_gem_drop_x_coordinates_for_npc_1()

        self.aquamarines = []
        self.amethysts = []
        self.emeralds = []
        self.rocks = []

        self.gem_lists = {
            "aquamarine": self.aquamarines,
            "amethyst": self.amethysts,
            "emerald": self.emeralds,
            "rock": self.rocks,
        }

        self.lava_list = []
        self.lava_dropped_on_traverse = False

        self.npc_2_gems = [self.amethysts, self.rocks]

        self.episode_step = 0
        self.miner_floor_sprite_indexes = [
            random.randint(0, 2) for _ in range(self.grid_side_length)
        ]
        self.floor_sprite_indexes = [random.randint(0, 2) for _ in range(self.grid_side_length)]

        observation = self._get_obs()

        info = {}
        return observation, info

    def _get_obs(self) -> np.ndarray:
        # Create a blank canvas (black) with shape (grid_side_length, grid_side_length, 3)
        obs = np.full(
            (self.grid_side_length, self.grid_side_length, 3),
            self.BACKGROUND_COLOR,
            dtype=np.uint8,
        )

        # --- Miner floor layers --
        for y in range(3):
            for x in range(self.grid_side_length):
                obs[y, x] = self.GROUND_COLOR_1 if y < 2 else self.GROUND_COLOR_2

        # --- Ground Layers ---
        # Fill the bottom row with GROUND_COLOR_1 and GROUND_COLOR_2
        for x in range(self.grid_side_length):
            obs[self.grid_side_length - 1, x] = self.GROUND_COLOR_2

        # --- Gems ---
        # Rocks
        for rock in self.rocks:
            obs[rock.y, rock.x] = self.ROCK_COLOR

        # Emeralds
        for emerald in self.emeralds:
            obs[emerald.y, emerald.x] = self.EMERALD_COLOR

        # Amethysts
        for amethyst in self.amethysts:
            obs[amethyst.y, amethyst.x] = self.AMETHYST_COLOR

        # Aquamarines
        for aquamarine in self.aquamarines:
            obs[aquamarine.y, aquamarine.x] = self.AQUAMARINE_COLOR

        # --- NPCs ---
        # NPC 1
        obs[self.npc_1.y, self.npc_1.x] = self.NPC_1_COLOR

        # NPC 2
        obs[self.npc_2.y, self.npc_2.x] = self.NPC_2_COLOR

        # --- Agent ---
        obs[self.agent.y, self.agent.x] = self.AGENT_COLOR

        # --- Lava ---
        for lava in self.lava_list:
            obs[lava.y, lava.x] = self.LAVA_COLOR

        return obs

    def _npc_1_drop_gems(self) -> None:
        if self.npc_1.x == 0 or self.npc_1.x == self.grid_side_length - 1:
            self.gems_dropped["aquamarine"] += 1
            self.aquamarines.append(
                Entity(grid_side_length=self.grid_side_length, starting_position=(self.npc_1.x, 0))
            )
        if self.npc_1.x in self.npc_1_gem_drop_coordinates.keys():
            gem_name = self.npc_1_gem_drop_coordinates[self.npc_1.x]
            self.gems_dropped[gem_name] += 1
            gem_list = self.gem_lists[gem_name]
            gem_list.append(
                Entity(grid_side_length=self.grid_side_length, starting_position=(self.npc_1.x, 0))
            )

    def _drop_lava(self):
        if self.lava_dropped_on_traverse == True:
            return
        # This condition ensures that agent won't be hit when following optimal policy
        if (self.npc_2.x == 10 and self.npc_2_moving_left == False) or (
            self.npc_2.x == 9 and self.npc_2_moving_left == True
        ):
            return
        if self.npc_2.x in self.npc_2_gem_drop_coordinates.keys():
            return
        if self.npc_2.x == 0 or self.npc_2.x == self.grid_side_length - 1:
            if random.random() > 0.5:
                return
        else:
            if random.random() > 0.1:
                return
        self.lava_list.append(
            Entity(grid_side_length=self.grid_side_length, starting_position=(self.npc_2.x, 0))
        )
        self.lava_dropped_on_traverse = True

    def _npc_2_drop_gems(self) -> None:
        if self.npc_2.x in self.npc_2_gem_drop_coordinates.keys():
            gem_name = self.npc_2_gem_drop_coordinates[self.npc_2.x]
            self.gems_dropped[gem_name] += 1
            gem_list = self.gem_lists[gem_name]
            gem_list.append(
                Entity(grid_side_length=self.grid_side_length, starting_position=(self.npc_2.x, 0))
            )

        # The code below shows that the agent can only pick up exactly ONE of the gems dropped by npc 2 every rotation, IF the optimal policy is followed
        """ if (self.npc_2.x == 10 and self.npc_2_moving_left == False) or (
            self.npc_2.x == 9 and self.npc_2_moving_left == True
        ):
            self.emeralds.append(
                Entity(grid_side_length=self.grid_side_length, starting_position=(self.npc_2.x, 0))
            )
        else:
            self.rocks.append(
                Entity(grid_side_length=self.grid_side_length, starting_position=(self.npc_2.x, 0))
            ) """

    def _update_gems_and_lava(self) -> None:

        for gem_list in [self.aquamarines, self.amethysts, self.emeralds, self.rocks]:
            for gem in gem_list[:]:
                if gem.y == self.grid_side_length - 1:
                    gem_list.remove(gem)
                else:
                    gem.action(3)  # Moves the gem 1 cell down

        for lava in self.lava_list[:]:
            if lava.y == self.grid_side_length - 1:
                self.lava_list.remove(lava)
            else:
                lava.action(3)

        self._npc_1_drop_gems()
        self._npc_2_drop_gems()
        self._drop_lava()

    def _npc_act(self) -> None:
        if self.npc_2_moving_left:
            self.npc_2.action(0)
            if self.npc_2.x == 0:
                self.npc_2_moving_left = False
                self.npc_2_gem_drop_coordinates = self._set_gem_drop_x_coordinates_for_npc_2()
                self.lava_dropped_on_traverse = False
        else:
            self.npc_2.action(1)
            if self.npc_2.x == self.grid_side_length - 1:
                self.npc_2_moving_left = True
                self.npc_2_gem_drop_coordinates = self._set_gem_drop_x_coordinates_for_npc_2()
                self.lava_dropped_on_traverse = False

        if self.npc_1_moving_left:
            self.npc_1.action(0)
            if self.npc_1.x == 0:
                self.npc_1_moving_left = False
                self.npc_1_gem_drop_coordinates = self._set_gem_drop_x_coordinates_for_npc_1()

        else:
            self.npc_1.action(1)
            if self.npc_1.x == self.grid_side_length - 1:
                self.npc_1_moving_left = True
                self.npc_1_gem_drop_coordinates = self._set_gem_drop_x_coordinates_for_npc_1()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:

        self.agent_collected = "none"
        self.agent.action(action)
        self._update_gems_and_lava()
        self._npc_act()

        if action == 0:
            self.agent_moving_left = True
        elif action == 1:
            self.agent_moving_left = False

        reward = 0
        terminated = False
        truncated = False

        for gem_type in self.gem_lists.keys():
            gem_list = self.gem_lists[gem_type]
            for gem in gem_list[:]:
                if gem == self.agent:
                    self.gems_collected[gem_type] += 1
                    gem_list.remove(gem)
                    reward += self.rewards[gem_type]
                    if gem_type == "rock":
                        self.agent_collected = "rock"
                    else:
                        self.agent_collected = "gem"

        for lava in self.lava_list:
            if lava == self.agent:
                reward += self.rewards["lava"]
                terminated = True

        new_observation = self._get_obs()

        if self.episode_step >= 190:  # max 10 rounds back and forth
            truncated = True
            terminated = True

        info = {}

        self.episode_step += 1

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
            try:
                aquamarine_path = resources.files("gem_collector.envs.sprites") / "aquamarine.png"
                self.aquamarine_sprite = pygame.image.load(str(aquamarine_path)).convert_alpha()
                amethyst_path = resources.files("gem_collector.envs.sprites") / "amethyst.png"
                self.amethyst_sprite = pygame.image.load(str(amethyst_path)).convert_alpha()
                emerald_path = resources.files("gem_collector.envs.sprites") / "emerald.png"
                self.emerald_sprite = pygame.image.load(str(emerald_path)).convert_alpha()
                rock_path = resources.files("gem_collector.envs.sprites") / "rock.png"
                self.rock_sprite = pygame.image.load(str(rock_path)).convert_alpha()
                lava_path = resources.files("gem_collector.envs.sprites") / "lava.png"
                self.lava_sprite = pygame.image.load(str(lava_path)).convert_alpha()
                agent_path = resources.files("gem_collector.envs.sprites") / "agent.png"
                self.agent_sprite = pygame.image.load(str(agent_path)).convert_alpha()
                agent_gem_collected_path = (
                    resources.files("gem_collector.envs.sprites") / "agent_gem_collected.png"
                )
                self.agent_gem_collected_sprite = pygame.image.load(
                    str(agent_gem_collected_path)
                ).convert_alpha()
                agent_rock_collected_path = (
                    resources.files("gem_collector.envs.sprites") / "agent_rock_collected.png"
                )
                self.agent_rock_collected_sprite = pygame.image.load(
                    str(agent_rock_collected_path)
                ).convert_alpha()

                npc_1_path = resources.files("gem_collector.envs.sprites") / "npc_1.png"
                self.npc_1_sprite = pygame.image.load(str(npc_1_path)).convert_alpha()
                npc_2_path = resources.files("gem_collector.envs.sprites") / "npc_2.png"
                self.npc_2_sprite = pygame.image.load(str(npc_2_path)).convert_alpha()

                floor_1_path = resources.files("gem_collector.envs.sprites") / "floor_1.png"
                self.floor_1_sprite = pygame.image.load(str(floor_1_path)).convert_alpha()
                floor_2_path = resources.files("gem_collector.envs.sprites") / "floor_2.png"
                self.floor_2_sprite = pygame.image.load(str(floor_2_path)).convert_alpha()
                floor_3_path = resources.files("gem_collector.envs.sprites") / "floor_3.png"
                self.floor_3_sprite = pygame.image.load(str(floor_3_path)).convert_alpha()
            except Exception as e:
                raise FileNotFoundError(f"Error loading sprite: {e}")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.show_raw_pixels == False:

            sprite_roatation_degrees = [0, 45, 90, 135, 180, 225, 270, 315]
            canvas = pygame.Surface((self.window_size, self.window_size))
            canvas.fill((77, 77, 77))
            pix_square_size = self.window_size / self.grid_side_length

            for combination in itertools.product(
                [n for n in range(self.grid_side_length)], [0, 1, 2]
            ):
                pygame.draw.rect(
                    canvas,
                    self.GROUND_COLOR_1,
                    pygame.Rect(
                        pix_square_size * combination[0],
                        pix_square_size * combination[1],
                        pix_square_size,
                        pix_square_size,
                    ),
                )

            for x in range(self.grid_side_length):
                # Randomly pick a floor sprite
                floor_sprite = [self.floor_1_sprite, self.floor_2_sprite, self.floor_3_sprite][
                    self.miner_floor_sprite_indexes[x]
                ]
                # Scale the sprite to match the grid cell
                scaled_sprite = pygame.transform.scale(
                    floor_sprite,
                    (int(pix_square_size), int(pix_square_size)),  # Resize to match a grid cell
                )

                # Blit the scaled sprite onto the canvas at the bottom row
                canvas.blit(
                    scaled_sprite,
                    (pix_square_size * x, pix_square_size * 2),
                )

            # floor
            # Draw the floor covering only the bottom row of the grid
            # Bottom row index
            for x in range(self.grid_side_length):
                # Randomly pick a floor sprite
                floor_sprite = [self.floor_1_sprite, self.floor_2_sprite, self.floor_3_sprite][
                    self.floor_sprite_indexes[x]
                ]
                # Scale the sprite to match the grid cell
                scaled_sprite = pygame.transform.scale(
                    floor_sprite,
                    (int(pix_square_size), int(pix_square_size)),  # Resize to match a grid cell
                )

                # Blit the scaled sprite onto the canvas at the bottom row
                canvas.blit(
                    scaled_sprite,
                    (pix_square_size * x, pix_square_size * (self.grid_side_length - 1)),
                )

            for rock in self.rocks:
                # Scale the sprite to the desired size
                scaled_sprite = pygame.transform.scale(
                    self.rock_sprite,
                    (int(pix_square_size), int(pix_square_size)),  # Resize to match a grid cell
                )

                # Rotate 180 degrees every episode step
                if self.episode_step % 2 == 1:  # Odd steps -> rotate 180 degrees
                    degree_i = random.randint(0, len(sprite_roatation_degrees) - 1)
                    scaled_sprite = pygame.transform.rotate(
                        scaled_sprite, sprite_roatation_degrees[degree_i]
                    )

                # Blit the (rotated) scaled sprite onto the canvas
                canvas.blit(
                    scaled_sprite,
                    (pix_square_size * rock.x, pix_square_size * rock.y),
                )

            for emerald in self.emeralds:
                # Scale the sprite to the desired size
                scaled_sprite = pygame.transform.scale(
                    self.emerald_sprite,
                    (int(pix_square_size), int(pix_square_size)),  # Resize to match a grid cell
                )

                # Rotate 180 degrees every episode step
                if self.episode_step % 2 == 1:  # Odd steps -> rotate 180 degrees
                    degree_i = random.randint(0, len(sprite_roatation_degrees) - 1)
                    scaled_sprite = pygame.transform.rotate(
                        scaled_sprite, sprite_roatation_degrees[degree_i]
                    )

                # Blit the (rotated) scaled sprite onto the canvas
                canvas.blit(
                    scaled_sprite,
                    (pix_square_size * emerald.x, pix_square_size * emerald.y),
                )

            for amethyst in self.amethysts:
                # Scale the sprite to the desired size
                scaled_sprite = pygame.transform.scale(
                    self.amethyst_sprite,
                    (int(pix_square_size), int(pix_square_size)),  # Resize to match a grid cell
                )

                # Rotate 180 degrees every episode step
                if self.episode_step % 2 == 1:  # Odd steps -> rotate 180 degrees
                    degree_i = random.randint(0, len(sprite_roatation_degrees) - 1)
                    scaled_sprite = pygame.transform.rotate(
                        scaled_sprite, sprite_roatation_degrees[degree_i]
                    )

                # Blit the (rotated) scaled sprite onto the canvas
                canvas.blit(
                    scaled_sprite,
                    (pix_square_size * amethyst.x, pix_square_size * amethyst.y),
                )

            for aquamarine in self.aquamarines:
                # Scale the sprite to the desired size
                scaled_sprite = pygame.transform.scale(
                    self.aquamarine_sprite,
                    (int(pix_square_size), int(pix_square_size)),  # Resize to match a grid cell
                )

                # Rotate 180 degrees every episode step
                if self.episode_step % 2 == 1:  # Odd steps -> rotate 180 degrees
                    degree_i = random.randint(0, len(sprite_roatation_degrees) - 1)
                    scaled_sprite = pygame.transform.rotate(
                        scaled_sprite, sprite_roatation_degrees[degree_i]
                    )

                # Blit the (rotated) scaled sprite onto the canvas
                canvas.blit(
                    scaled_sprite,
                    (pix_square_size * aquamarine.x, pix_square_size * aquamarine.y),
                )

            # Drawing the npc 1
            scaled_sprite = pygame.transform.scale(
                self.npc_1_sprite,
                (int(pix_square_size), int(pix_square_size)),  # Resize to match a grid cell
            )

            if self.npc_1_moving_left:
                scaled_sprite = pygame.transform.flip(
                    scaled_sprite, True, False
                )  # Flip horizontally

            canvas.blit(
                scaled_sprite,
                (pix_square_size * self.npc_1.x, pix_square_size * self.npc_1.y),
            )

            # Drawing the npc 2
            scaled_sprite = pygame.transform.scale(
                self.npc_2_sprite,
                (int(pix_square_size), int(pix_square_size)),  # Resize to match a grid cell
            )

            if self.npc_2_moving_left:
                scaled_sprite = pygame.transform.flip(
                    scaled_sprite, True, False
                )  # Flip horizontally

            canvas.blit(
                scaled_sprite,
                (pix_square_size * self.npc_2.x, pix_square_size * self.npc_2.y),
            )

            # Drawing the agent
            if self.agent_collected == "rock":
                agent_sprite = self.agent_rock_collected_sprite
            elif self.agent_collected == "gem":
                agent_sprite = self.agent_gem_collected_sprite
            else:
                agent_sprite = self.agent_sprite

            scaled_sprite = pygame.transform.scale(
                agent_sprite,
                (int(pix_square_size), int(pix_square_size)),  # Resize to match a grid cell
            )

            if self.agent_moving_left:
                scaled_sprite = pygame.transform.flip(
                    scaled_sprite, True, False
                )  # Flip horizontally

            canvas.blit(
                scaled_sprite,
                (pix_square_size * self.agent.x, pix_square_size * self.agent.y),
            )

            for lava in self.lava_list:
                # Scale the sprite to the desired size
                scaled_sprite = pygame.transform.scale(
                    self.lava_sprite,
                    (int(pix_square_size), int(pix_square_size)),  # Resize to match a grid cell
                )

                if self.episode_step % 2 == 1:  # Odd steps -> flip horizontally
                    scaled_sprite = pygame.transform.flip(
                        scaled_sprite, True, False
                    )  # Flip horizontally

                canvas.blit(
                    scaled_sprite,
                    (pix_square_size * lava.x, pix_square_size * lava.y),
                )

        else:
            obs_grid = self._get_obs()

            # Create a surface to visualize the obs grid
            canvas = pygame.Surface((self.window_size, self.window_size))
            pix_square_size = self.window_size / self.grid_side_length

            # Loop through each grid cell in obs and draw its corresponding color
            for y in range(self.grid_side_length):
                for x in range(self.grid_side_length):
                    color = tuple(obs_grid[y, x])  # Extract the RGB color from the obs array
                    pygame.draw.rect(
                        canvas,
                        color,
                        pygame.Rect(
                            x * pix_square_size,  # X-position
                            y * pix_square_size,  # Y-position
                            pix_square_size,  # Width
                            pix_square_size,  # Height
                        ),
                    )

            # Blit the canvas to the window and update display
            self.window.blit(canvas, (0, 0))
            pygame.display.update()
            self.clock.tick(self.render_fps)

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.render_fps)

    def _get_ground_color(self, x: int, y: int) -> Tuple[int, int, int]:
        if y == self.grid_side_length - 2:
            return self.GROUND_COLOR_1 if x % 2 == 0 else self.GROUND_COLOR_2
        else:
            return self.GROUND_COLOR_2 if x % 2 == 0 else self.GROUND_COLOR_1
