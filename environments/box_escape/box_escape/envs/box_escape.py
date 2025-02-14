import random
from typing import Any, Dict, List, Set, SupportsFloat, Tuple, Type

import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv

from .actions import Actions
from .box import Box
from .number_tile import NumberTile
from .static_obj_map import STATIC_OBJ_MAP


class BoxEscape(MiniGridEnv):
    """
    A custom MiniGrid environment where the agent must navigate around boxes,
    pick up the key, and unlock a door to reach the goal. There are four doors,
    and the agent receives a reward based on whether it has entered the correct
    door or not. The correct door can be found by matching the number at each
    door with the number of boxes with the same color as the doors and the key.

    The agent moves in a discrete grid-based environment and can interact with
    objects such as doors, keys, and boxes. The objective is to determine the
    correct goal door using the number clues and successfully reach it to maximize
    rewards. The environment supports both **fully observable** and **partially observable**
    settings.

    The agent has a limited set of **discrete actions**, including:
    - Moving forward
    - Rotating left or right
    - Picking up and dropping objects
    - Toggling/interacting with doors
    """

    # Rewards so that for each curriculum level, the episode reward range is [-1,1]
    # Max steps = 200, so 200 * -0.005 = -1
    # Min steps to collect key = 1 (for all types of env instances)
    # Min steps to collect key, unlock door and go to goal = 6 (for all types of env instances)
    REWARDS: Dict[int, Dict[str, float]] = {
        1: {
            "key_pickup": 1.001,
            "pickup_penalty": -0.004,
            "toggle_penalty": -0.004,
            "correct_goal_reached": 0,
            "incorrect_goal_reached": 0,
            "step_penalty": -0.001,
        },
        2: {
            "key_pickup": 1.001,
            "pickup_penalty": -0.004,
            "toggle_penalty": -0.004,
            "correct_goal_reached": 0,
            "incorrect_goal_reached": 0,
            "step_penalty": -0.001,
        },
        3: {
            "key_pickup": 0.3,
            "pickup_penalty": -0.004,
            "toggle_penalty": -0.004,
            "correct_goal_reached": 0.706,
            "incorrect_goal_reached": 0.706,
            "step_penalty": -0.001,
        },
        4: {
            "key_pickup": 0.2,
            "pickup_penalty": -0.004,
            "toggle_penalty": -0.004,
            "correct_goal_reached": 0.806,
            "incorrect_goal_reached": 0,
            "step_penalty": -0.001,
        },
    }
    VALID_OBJ_COLOR_INDEXES: List[int] = [0, 1, 3, 4, 5]  # Not including 2, which is gray
    GOAL_POS_MAP: Dict[Tuple[int, int], int] = {
        (7, 1): 1,
        (13, 7): 2,
        (7, 13): 3,
        (1, 7): 4,
    }
    MAX_STEPS: int = 200
    VALID_CURRICULUM_LEVELS = [1, 2, 3, 4]

    def __init__(
        self,
        fully_observable: bool = True,
        curriculum_level: int = 4,
        **kwargs,
    ):

        mission_space: MissionSpace = MissionSpace(mission_func=self._gen_mission)
        self.size: int = 15
        self.fully_observable: bool = fully_observable

        if curriculum_level not in self.VALID_CURRICULUM_LEVELS:
            raise ValueError(
                f"Invalid curriculum level. Must be one of {self.VALID_CURRICULUM_LEVELS}"
            )

        self.curriculum_level = curriculum_level

        super().__init__(
            mission_space=mission_space,
            grid_size=self.size,
            see_through_walls=True,
            max_steps=self.MAX_STEPS,
            **kwargs,
        )

        self.actions: Type[Actions] = Actions
        self.action_space: spaces.Discrete = spaces.Discrete(len(self.actions))

    @staticmethod
    def _gen_mission():
        return ""

    def _place_static_objects(self) -> List[Tuple[int, int]]:
        empty_cells: List[Tuple[int, int]] = []
        for y in range(self.size):
            for x in range(self.size):
                coord_value: int = STATIC_OBJ_MAP[y][x]
                if coord_value == 1:
                    self.grid.set(x, y, Wall())
                elif coord_value == 2:
                    self.put_obj(Goal(), x, y)
                elif coord_value == 3:
                    door = Door(COLOR_NAMES[self.chosen_color_i], is_locked=True)
                    self.grid.set(x, y, door)
                elif coord_value == 0:
                    empty_cells.append((x, y))
        return empty_cells

    def _place_key(
        self, empty_cells: List[Tuple[int, int]]
    ) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
        key_pos: Tuple[int, int] = random.choice(empty_cells)
        self.grid.set(*key_pos, Key(COLOR_NAMES[self.chosen_color_i]))
        empty_cells.remove(key_pos)
        return empty_cells, key_pos

    def _place_agent(
        self, empty_cells: List[Tuple[int, int]], key_pos: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        while True:
            agent_pos: Tuple[int, int] = random.choice(empty_cells)
            # Ensure the agent is at least two squares away from the key
            if abs(agent_pos[0] - key_pos[0]) + abs(agent_pos[1] - key_pos[1]) >= 2:
                self.agent_pos = agent_pos
                self.agent_dir = random.randint(0, 3)
                self.grid.set(*self.agent_pos, None)
                empty_cells.remove(self.agent_pos)
                return empty_cells

    def _place_boxes(self, empty_cells: List[Tuple[int, int]]) -> None:
        for color_i in self.VALID_OBJ_COLOR_INDEXES:
            boxes_n: int = (
                self.target_direction if color_i == self.chosen_color_i else random.randint(1, 4)
            )
            first_box_pos: Tuple[int, int] = random.choice(empty_cells)
            self.put_obj(Box(COLOR_NAMES[color_i]), *first_box_pos)
            empty_cells.remove(first_box_pos)

            placed_boxes: List[Tuple[int, int]] = [first_box_pos]
            while len(placed_boxes) < boxes_n:
                adjacent_positions = set()
                for x, y in placed_boxes:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, down, left, right
                        nx, ny = x + dx, y + dy
                        if (nx, ny) in empty_cells:
                            adjacent_positions.add((nx, ny))

                # If no valid adjacent position exists, stop placement
                if not adjacent_positions:
                    break

                # Choose a random position from the available adjacent positions
                next_box_pos = random.choice(list(adjacent_positions))
                self.put_obj(Box(COLOR_NAMES[color_i]), *next_box_pos)
                empty_cells.remove(next_box_pos)
                placed_boxes.append(next_box_pos)

    def _place_number_tiles(self) -> None:
        self.grid.set(7, 0, NumberTile(color="blue"))
        self.grid.set(14, 7, NumberTile(color="red"))
        self.grid.set(7, 14, NumberTile(color="green"))
        self.grid.set(0, 7, NumberTile(color="yellow"))

    def _gen_grid(self, width, height):
        self.chosen_color_i: int = random.choice(
            self.VALID_OBJ_COLOR_INDEXES
        )  # Not including 2, which is gray
        self.target_direction: int = random.randint(1, 4)
        self.grid: Grid = Grid(width, height)
        empty_cells: List[Tuple[int, int]] = self._place_static_objects()
        if self.curriculum_level > 1:
            self._place_boxes(empty_cells=empty_cells)
        empty_cells, key_pos = self._place_key(empty_cells=empty_cells)
        empty_cells: List[Tuple[int, int]] = self._place_agent(
            empty_cells=empty_cells, key_pos=key_pos
        )
        self._place_number_tiles()

        self.mission = None

    def gen_obs_grid(self, agent_view_size=None):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        if agent_view_size is None, self.agent_view_size is used
        """

        topX, topY, _, _ = self.get_view_exts(agent_view_size)

        agent_view_size = agent_view_size or self.agent_view_size

        grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)

        for _ in range(self.agent_dir + 1):
            grid = grid.rotate_left()

        # Makes the mask transparent, since it is not needed in the observations of the entire env
        if self.fully_observable:
            vis_mask = np.zeros(shape=(grid.width, grid.height), dtype=bool)

        # Process occluders and visibility
        # Note that this incurs some performance cost
        elif not self.see_through_walls:
            vis_mask = grid.process_vis(agent_pos=(agent_view_size // 2, agent_view_size - 1))
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height - 1
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                pos_converted = (int(fwd_pos[0]), int(fwd_pos[1]))
                goal_reached = self.GOAL_POS_MAP.get(pos_converted)
                if goal_reached == self.target_direction:
                    reward = self.REWARDS[self.curriculum_level]["correct_goal_reached"]
                else:
                    reward = self.REWARDS[self.curriculum_level]["incorrect_goal_reached"]

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)
                    if fwd_cell.type == "key":
                        reward = self.REWARDS[self.curriculum_level]["key_pickup"]
                        if self.curriculum_level < 3:
                            terminated = True
            else:
                reward = self.REWARDS[self.curriculum_level]["pickup_penalty"]

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                can_toggle = fwd_cell.toggle(self, fwd_pos)
                if not can_toggle:
                    reward = self.REWARDS[self.curriculum_level]["toggle_penalty"]
            else:
                reward = self.REWARDS[self.curriculum_level]["toggle_penalty"]

        else:
            raise ValueError(f"Unknown action: {action}")

        reward += self.REWARDS[self.curriculum_level]["step_penalty"]

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}
