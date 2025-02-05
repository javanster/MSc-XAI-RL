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
    """
    A Gymnasium environment for the Gem Collector game, where an agent collects gems while avoiding obstacles.

    GemCollector has exactly one unique optimal policy, consisting of continuously traversing the full length of
    the environment back and forth, thereby collecting every aquamarine and amethyst.

    Attributes
    ----------
    metadata : dict
        Dictionary containing rendering metadata, such as available render modes and frames per second.
    name : str
        Name of the environment.
    reward_range : Tuple[int, int]
        The minimum and maximum possible rewards.
    grid_side_length : int
        The size of the grid environment.
    action_space : Discrete
        The action space for the agent, consisting of three actions: move left, move right, and do nothing.
    action_dict : Dict[int, str]
        Mapping of action indices to corresponding movement directions.
    observation_space : Box
        The observation space representing the environment's state as an RGB image.
    SPRITE_MODULE_PATH : str
        Path to the sprite assets for rendering.
    entity_colors : Dict[str, Tuple[int, int, int]]
        Mapping of entity names to their RGB color representations.
    show_raw_pixels : bool
        Flag indicating whether to show raw pixel values during rendering.
    window_size : int
        The size of the rendering window in pixels.
    render_fps : int
        Frames per second for rendering.
    render_mode : Optional[str]
        The mode of rendering, either "human" or "rgb_array".
    window : Optional[pygame.Surface]
        The Pygame window surface for rendering.
    clock : Optional[pygame.time.Clock]
        The Pygame clock for controlling rendering speed.
    rewards : Dict[str, float]
        A dictionary mapping collectible objects to their respective reward values.
    obj_lists : Dict[str, List[Entity]]
        Dictionary holding lists of collectible objects and obstacles.
    agent : Miner
        The agent controlled by the player.
    npc_1 : Miner
        The first non-player character (NPC) miner.
    npc_2 : Miner
        The second non-player character (NPC) miner.
    miner_floor_sprite_indexes : List[int]
        Indices for selecting random floor sprites for NPCs.
    floor_sprite_indexes : List[int]
        Indices for selecting random floor sprites for rendering.
    active_agent_sprite : int
        The current sprite index for the agent.
    episode_step : int
        Counter tracking the number of steps taken in the current episode.

    Methods
    -------
    reset(seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[Any, Any]]
        Resets the environment and returns the initial observation and metadata.
    step(action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]
        Updates the environment state based on the given action, returning the new observation, reward, termination and truncation flags, and additional metadata.
    render() -> None
        Renders the environment visually using Pygame.
    close() -> None
        Closes the rendering window and releases resources.
    _get_obs() -> np.ndarray
        Returns the current observation as a NumPy array.
    _is_entity_next_to_wall_x(entity: Entity) -> bool
        Checks if a given entity is adjacent to a wall.
    _drop_obj(entity_list: List[Entity], x_cord: int, entity_color: Tuple[int, int, int]) -> None
        Drops an object at a specified location.
    _gravitate_objects() -> None
        Moves gems, rocks and lava downward to simulate gravity.
    _npc_1_drop_objects() -> None
        Handles object-dropping behavior for NPC 1.
    _npc_2_drop_objects() -> None
        Handles object-dropping behavior for NPC 2.
    _move_npc(npc: Miner, obj_x_coord_reset: Callable[[], Dict[int, str]]) -> None
        Moves an NPC and resets its object drop positions if necessary.
    _load_sprite(sprite_name: str) -> pygame.Surface
        Loads a sprite image from the resources.
    _scale_and_blit_sprite(sprite: pygame.Surface, canvas: pygame.Surface, pix_square_size: float, x: int, y: int, random_rotate: bool = False, flip_horizontally: bool = False) -> None
        Scales and blits a sprite onto the canvas.
    _draw_sprites(canvas: pygame.Surface, pix_square_size: float) -> None
        Draws all sprites onto the canvas.
    _draw_entities() -> None
        Renders all entities and updates the display.

    Notes
    -----
    - The game is grid-based, where the agent and NPCs move left or right while objects fall downward.
    - The agent collects different types of gems for rewards but must avoid hazards like lava.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: Optional[LiteralString] = None,
        render_fps: int = 4,
        show_raw_pixels=False,
    ) -> None:
        super().__init__()
        # ----- METADATA -----
        self.name: str = "gem_collector"
        self.reward_range: Tuple[int, int] = (-1, 1)
        self.grid_side_length: int = 20
        self.action_space: Discrete = Discrete(3)
        self.action_dict: Dict[int, str] = {
            0: "left",
            1: "right",
            2: "do_nothing",
        }
        self.observation_space: Box = Box(
            low=0,
            high=255,
            shape=(self.grid_side_length, self.grid_side_length, 3),
            dtype=np.uint8,
        )
        self.SPRITE_MODULE_PATH: str = "gem_collector.envs.sprites"

        # ----- ENTITY COLORS -----
        self.entity_colors: Dict[str, Tuple[int, int, int]] = {
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
        self.show_raw_pixels: bool = show_raw_pixels
        self.window_size: int = 900
        self.render_fps: int = render_fps
        if render_mode and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Argument for 'render_mode' must either be 'None' or one of the following strings: {[rm for rm in self.metadata['render_modes']]}"
            )
        self.render_mode: Optional[str] = render_mode
        self.window: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None

        # ----- REWARDS -----
        self.rewards: Dict[str, float] = {
            "aquamarine": 0.045,
            "amethyst": 0.033,
            "emerald": 0.022,
            "rock": -0.0004,
            "lava": -0.996,
        }

    def _reset_and_get_obj_drop_x_coordinates_for_npc_1(self) -> Dict[int, str]:
        """
        Resets and determines the x-coordinate for NPC 1 to drop an amethyst.

        Returns
        -------
        Dict[int, str]
            A dictionary with a single key-value pair where the key is the randomly selected
            x-coordinate for NPC 1 to drop an "amethyst" and the value is the string "amethyst".

        Notes
        -----
        - The x-coordinate is randomly chosen within the range [1, grid_side_length - 2].
        - If the chosen x-coordinate is already used by NPC 2 for dropping an object, a new coordinate is selected.
        """
        amethyst_x_coordinates: int = random.randint(1, self.grid_side_length - 2)
        while amethyst_x_coordinates in self.npc_2.obj_drop_x_coordinates.keys():
            amethyst_x_coordinates: int = random.randint(1, self.grid_side_length - 2)
        return {amethyst_x_coordinates: "amethyst"}

    def _reset_and_get_obj_drop_x_coordinates_for_npc_2(self) -> Dict[int, str]:
        """
        Resets and determines the x-coordinates for NPC 2 to drop emeralds, rocks, and lava.

        Returns
        -------
        Dict[int, str]
            A dictionary where:
            - One randomly chosen x-coordinate is assigned to drop an "emerald".
            - Another randomly chosen x-coordinate is assigned to drop a "rock".
            - A third x-coordinate is assigned to drop "lava", ensuring it does not overlap
              with the emerald or rock coordinates and considers NPC 2's movement direction.

        Notes
        -----
        - The emerald and rock x-coordinates are selected randomly from the range [1, grid_side_length - 2].
        - The lava coordinate is determined with additional constraints:
            - It is randomly placed at the left or right boundary with 50% probability.
            - If placed elsewhere, it must not overlap with other dropped objects.
            - The placement also accounts for NPC 2's movement direction and x coordinate, to ensure exactly 1
              optimal policy.
        """
        emerald_rock_x_coordinates: List[int] = random.sample(
            range(1, self.grid_side_length - 1), 2
        )
        if random.random() > 0.5:
            lava_coordinate: int = random.randint(0, 1)
        else:
            lava_coordinate: int = random.randint(1, self.grid_side_length - 2)
            while (
                lava_coordinate in emerald_rock_x_coordinates
                or (lava_coordinate == 10 and not self.npc_2.is_moving_left)
                or (lava_coordinate == 9 and self.npc_2.is_moving_left)
            ):
                lava_coordinate: int = random.randint(1, self.grid_side_length - 2)
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
        """
        Resets the environment to its initial state and returns the first observation.

        This method reinitializes the agent, NPCs, and objects to their starting positions.
        It also randomly determines where NPCs will drop objects and resets the step counter.
        Rendering variables are reset as well, ensuring consistency at the start of each episode.

        Parameters
        ----------
        seed : Optional[int], default=None
            Seed for the random number generator to ensure reproducibility.
        options : Optional[Dict[str, Any]], default=None
            Additional options for customization (not explicitly used in this implementation).

        Returns
        -------
        Tuple[np.ndarray, Dict[Any, Any]]
            - The first observation of the environment as a NumPy array of shape
              (grid_side_length, grid_side_length, 3), representing the RGB pixel values.
            - An empty dictionary containing additional information (can be expanded in future implementations).
        """
        super().reset(seed=seed)

        # ----- RESETTING ENTITIES -----
        self.agent: Miner = Miner(
            grid_side_length=self.grid_side_length,
            starting_position=(0, self.grid_side_length - 2),
            color=self.entity_colors["agent"],
            is_moving_left=False,
        )
        self.npc_2: Miner = Miner(
            grid_side_length=self.grid_side_length,
            starting_position=(1, 1),
            color=self.entity_colors["npc_2"],
            is_moving_left=False,
        )
        self.npc_1: Miner = Miner(
            grid_side_length=self.grid_side_length,
            starting_position=(self.grid_side_length - 1, 1),
            color=self.entity_colors["npc_1"],
            is_moving_left=True,
        )
        self.npc_2.obj_drop_x_coordinates = self._reset_and_get_obj_drop_x_coordinates_for_npc_2()
        self.npc_1.obj_drop_x_coordinates = self._reset_and_get_obj_drop_x_coordinates_for_npc_1()

        self.obj_lists: Dict[str, List[Entity]] = {
            "aquamarine": [],
            "amethyst": [],
            "emerald": [],
            "rock": [],
            "lava": [],
        }

        # ----- RENDERING VARS -----
        self.miner_floor_sprite_indexes: List[int] = [
            random.randint(0, 2) for _ in range(self.grid_side_length)
        ]
        self.floor_sprite_indexes: List[int] = [
            random.randint(0, 2) for _ in range(self.grid_side_length)
        ]
        self.active_agent_sprite: int = 0

        self.episode_step: int = 0
        observation: np.ndarray = self._get_obs()
        info = {}
        return observation, info

    def _get_obs(self) -> np.ndarray:
        """
        Constructs and returns the current observation of the environment as an RGB image.

        The observation is represented as a NumPy array where each pixel corresponds to an entity
        in the environment. Background elements such as walls and floors are set first, followed
        by gems, the agent, and NPCs. Objects like lava, which affect gameplay, are also included
        in the observation.

        Returns
        -------
        np.ndarray
            A NumPy array of shape (grid_side_length, grid_side_length, 3), where each element
            represents an RGB color corresponding to the environment's current state.
        """
        # ----- BACKGROUND PIXELS -----
        obs: np.ndarray = np.full(
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
        """
        Checks whether a given entity is adjacent to a wall along the x-axis.

        This method determines if an entity is positioned at the leftmost or rightmost
        boundary of the grid.

        Parameters
        ----------
        entity : Entity
            The entity whose position is being evaluated.

        Returns
        -------
        bool
            True if the entity is at either the leftmost (x = 0) or rightmost
            (x = grid_side_length - 1) position, otherwise False.
        """
        return entity.x == 0 or entity.x == self.grid_side_length - 1

    def _drop_obj(
        self, entity_list: List[Entity], x_cord: int, entity_color: Tuple[int, int, int]
    ) -> None:
        """
        Drops an object at the specified x-coordinate and adds it to the corresponding entity list.

        This method creates a new entity (such as a gem, rock, or lava) at the given x-coordinate
        and places it at the top of the grid (y = 0). The new entity is then appended to the
        provided list, allowing it to be tracked and affected by gravity.

        Parameters
        ----------
        entity_list : List[Entity]
            The list to which the newly created entity will be added.
        x_cord : int
            The x-coordinate where the object will be dropped.
        entity_color : Tuple[int, int, int]
            The RGB color representing the entity.

        Returns
        -------
        None
        """
        entity_list.append(
            Entity(
                grid_side_length=self.grid_side_length,
                starting_position=(x_cord, 0),
                color=entity_color,
            )
        )

    def _gravitate_objects(self) -> None:
        """
        Moves all objects downward to simulate gravity.

        This method iterates through all tracked objects (such as gems, rocks, and lava)
        and moves them one cell downward if they are not already on the ground. Objects
        that reach the bottom of the grid are removed from their respective lists. This
        process ensures that collectible items and hazards behave naturally within the
        environment.

        Returns
        -------
        None
        """
        for entity_list in self.obj_lists.values():
            for entity in entity_list[:]:
                if entity.y == self.grid_side_length - 1:  # Entity is on the ground
                    entity_list.remove(entity)
                else:
                    entity.action(3)  # Moves the entity 1 cell down

    def _npc_1_drop_objects(self) -> None:
        """
        Handles the object-dropping behavior for NPC 1.

        NPC 1 drops objects based on its current x-coordinate. If it is adjacent to a
        wall, it will always drop an aquamarine. Otherwise, it checks its predefined
        drop coordinates and places the corresponding gem at that location. The dropped
        object is then added to the appropriate list for tracking.

        Returns
        -------
        None
        """
        if self._is_entity_next_to_wall_x(entity=self.npc_1):
            self._drop_obj(
                entity_list=self.obj_lists["aquamarine"],
                x_cord=self.npc_1.x,
                entity_color=self.entity_colors["aquamarine"],
            )
        elif self.npc_1.x in self.npc_1.obj_drop_x_coordinates.keys():
            gem_to_drop_name: str = self.npc_1.obj_drop_x_coordinates[self.npc_1.x]
            gem_list: List[Entity] = self.obj_lists[gem_to_drop_name]
            gem_color: Tuple[int, int, int] = self.entity_colors[gem_to_drop_name]
            self._drop_obj(
                entity_list=gem_list,
                x_cord=self.npc_1.x,
                entity_color=gem_color,
            )

    def _npc_2_drop_objects(self) -> None:
        """
        Handles the object-dropping behavior for NPC 2.

        NPC 2 drops objects based on its predefined x-coordinates for emeralds, rocks,
        and lava. If NPC 2 reaches an x-coordinate where an object is designated to be
        dropped, it will create the corresponding entity and add it to the appropriate
        list for tracking. This ensures that obstacles and collectible items are
        dynamically introduced into the environment.

        Returns
        -------
        None
        """
        if self.npc_2.x in self.npc_2.obj_drop_x_coordinates.keys():
            obj_to_drop_name: str = self.npc_2.obj_drop_x_coordinates[self.npc_2.x]
            obj_list: List[Entity] = self.obj_lists[obj_to_drop_name]
            obj_color: Tuple[int, int, int] = self.entity_colors[obj_to_drop_name]
            self._drop_obj(entity_list=obj_list, x_cord=self.npc_2.x, entity_color=obj_color)

    def _move_npc(self, npc: Miner, obj_x_coord_reset: Callable[[], Dict[int, str]]) -> None:
        """
        Moves an NPC in its current direction and resets object drop positions if necessary.

        This method determines the NPC's movement based on its current direction. If the
        NPC reaches a wall, its movement direction is reversed, and its object drop
        positions are reset using the provided function. This ensures that NPCs follow
        a predictable back-and-forth movement pattern while dynamically adjusting where
        they drop objects.

        Parameters
        ----------
        npc : Miner
            The NPC to be moved.
        obj_x_coord_reset : Callable[[], Dict[int, str]]
            A function that resets and returns the NPC's new object drop coordinates.

        Returns
        -------
        None
        """
        action: int = 0 if npc.is_moving_left else 1
        npc.action(action)
        if self._is_entity_next_to_wall_x(entity=npc):
            npc.is_moving_left = not npc.is_moving_left
            npc.obj_drop_x_coordinates = obj_x_coord_reset()

    def _get_rewards_obj_collision(self) -> Tuple[float, bool]:
        """
        Checks for collisions between the agent and objects, updating rewards and termination status.

        This method iterates through all objects in the environment and checks if the agent
        has collided with any of them. If a collision occurs, the object is removed from its
        respective list, and the agent receives a reward based on the object type. Certain
        objects, like rocks and gems, modify the agent’s sprite, while collisions with lava
        result in immediate termination of the episode.

        Returns
        -------
        Tuple[float, bool]
            - A float representing the accumulated reward from object collisions.
            - A boolean indicating whether the episode should terminate due to a lava collision.
        """
        reward: float = 0
        terminated = False
        for obj_type in self.obj_lists.keys():
            obj_list = self.obj_lists[obj_type]
            for obj in obj_list[:]:
                if obj == self.agent:
                    reward += self.rewards[obj_type]
                    if obj_type == "rock":
                        self.active_agent_sprite = 1
                    elif obj_type == "lava":
                        terminated = True
                        break
                    else:
                        self.active_agent_sprite = 2
                    obj_list.remove(obj)
        return reward, terminated

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        """
        Executes a single step in the environment based on the agent's action.

        This method updates the agent's position based on the given action, applies gravity to objects,
        handles NPC movements and object drops, and checks for collisions between the agent and objects.
        The reward and termination status are updated accordingly. If the maximum number of steps
        (190) is reached, the episode is truncated and terminated. The updated observation is returned
        along with the new environment state.

        Parameters
        ----------
        action : int
            The action taken by the agent.
            - 0: Move left
            - 1: Move right
            - 2: Do nothing

        Returns
        -------
        Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]
            - A NumPy array representing the new observation of the environment.
            - A float representing the reward earned from the step.
            - A boolean indicating whether the episode has terminated (e.g., due to collision with lava).
            - A boolean indicating whether the episode was truncated due to reaching the step limit.
            - A dictionary containing additional environment information (currently empty).
        """
        self.active_agent_sprite: int = 0
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

        reward, terminated = self._get_rewards_obj_collision()
        truncated: bool = False

        if self.episode_step >= 190:  # max 10 complete rounds back and forth
            truncated = True
            terminated = True

        info = {}
        self.episode_step += 1
        new_observation: np.ndarray = self._get_obs()

        return new_observation, reward, terminated, truncated, info

    def render(self) -> None:
        """
        Renders the current state of the environment based on the selected render mode.

        If the render mode is set to "human", the method calls `_draw_entities()` to
        visually display the environment using Pygame.

        Returns
        -------
        None
        """
        if self.render_mode == "human":
            self._draw_entities()

    def close(self) -> None:
        """
        Closes the rendering window and releases associated resources.

        This method ensures that Pygame's display and system resources are properly
        cleaned up when the environment is no longer in use. It quits the Pygame display
        if a window has been created.

        Returns
        -------
        None
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _load_sprite(self, sprite_name: str) -> pygame.Surface:
        """
        Loads a sprite image from the specified resources path.

        This method retrieves a sprite file from the environment's sprite module, loads it
        as a Pygame surface, and ensures it supports transparency using `convert_alpha()`.
        The sprite is used for rendering various game elements like the agent, NPCs, and objects.

        Parameters
        ----------
        sprite_name : str
            The name of the sprite file (without the file extension) to be loaded.

        Returns
        -------
        pygame.Surface
            A Pygame surface object representing the loaded sprite.
        """
        path: Traversable = resources.files(self.SPRITE_MODULE_PATH) / f"{sprite_name}.png"
        return pygame.image.load(str(path)).convert_alpha()

    def _scale_and_blit_sprite(
        self,
        sprite: pygame.Surface,
        canvas: pygame.Surface,
        pix_square_size: float,
        x: int,
        y: int,
        random_rotate: bool = False,
        flip_horizontally: bool = False,
    ) -> None:
        """
        Scales, transforms, and blits a sprite onto the rendering canvas.

        This method resizes a given sprite to match the grid cell size, optionally applies
        random rotation and horizontal flipping, and then places it at the specified
        coordinates on the canvas. It is used to render game entities such as the agent,
        NPCs, and objects while maintaining visual consistency.

        Parameters
        ----------
        sprite : pygame.Surface
            The Pygame surface representing the sprite to be drawn.
        canvas : pygame.Surface
            The rendering surface where the sprite will be blitted.
        pix_square_size : float
            The size of a single grid cell in pixels.
        x : int
            The x-coordinate on the grid where the sprite will be placed.
        y : int
            The y-coordinate on the grid where the sprite will be placed.
        random_rotate : bool, default=False
            If True, the sprite will be randomly rotated by a multiple of 45 degrees.
        flip_horizontally : bool, default=False
            If True, the sprite will be flipped horizontally before being drawn.

        Returns
        -------
        None
        """
        sprite_roatation_degrees = [d for d in range(0, 316, 45)]
        scaled_sprite: pygame.Surface = pygame.transform.scale(
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

    def _draw_sprites(self, canvas: pygame.Surface, pix_square_size: float) -> None:
        """
        Draws all sprites onto the rendering canvas.

        This method loads and places the appropriate sprites for walls, floors, objects,
        the agent, NPCs, and other entities in the environment. It ensures that the
        visual representation of the game is correctly displayed using scaled and
        transformed sprite images. The method uses predefined mappings to determine
        which sprites to use for each entity and their respective positions.

        Parameters
        ----------
        canvas : pygame.Surface
            The rendering surface where the sprites will be drawn.
        pix_square_size : float
            The size of a single grid cell in pixels, used to properly scale and position sprites.

        Returns
        -------
        None
        """
        sprites: Dict[str, pygame.Surface] = {
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
        """
        Handles the rendering of all entities and updates the display.

        This method initializes the Pygame display if it hasn't been set up, creates a
        canvas for rendering, and determines whether to display raw pixel data or use
        sprites for a more detailed visual representation. It then blits the final
        rendering onto the display window and updates the screen while maintaining
        the specified frame rate.

        Returns
        -------
        None
        """
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas: pygame.Surface = pygame.Surface((self.window_size, self.window_size))
        pix_square_size: float = self.window_size / self.grid_side_length

        if self.show_raw_pixels:
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

        self.window.blit(canvas, canvas.get_rect())  # type: ignore
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.render_fps)  # type: ignore
