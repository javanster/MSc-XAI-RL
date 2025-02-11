import random
from typing import Any, Dict, List, Set, SupportsFloat, Tuple, Type

import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Box, Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv

from .actions import Actions
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

    The **reward function** is structured as follows:
    - A reward is given for picking up the key.
    - A higher reward is given for reaching the correct goal.
    - A lower reward is given for reaching an incorrect goal.
    - The sum of `"key_pickup"` and `"correct_goal_reached"` must always be `1.0`.

    The environment is dynamically generated based on a predefined **static object map**,
    ensuring structured levels while maintaining some randomness in object placement.

    Parameters
    ----------
    rewards : dict of str to float, optional
        A dictionary specifying reward values for:
        - "key_pickup": Reward for picking up the key.
        - "correct_goal_reached": Reward for entering the correct door.
        - "incorrect_goal_reached": Reward for entering an incorrect door.
        The dictionary must contain exactly these keys with valid values.
    fully_observable : bool, optional
        If True, the agent receives full environment observations instead of partial views.
    **kwargs : dict
        Additional arguments passed to the MiniGridEnv constructor.

    Attributes
    ----------
    REQUIRED_REWARD_KEYS : set of str
        The required keys for the rewards dictionary.
    VALID_OBJ_COLOR_INDEXES : list of int
        List of valid object color indices, excluding gray.
    GOAL_POS_MAP : dict of tuple to int
        Mapping of goal positions to target directions.
    MAX_STEPS : int
        The maximum number of steps allowed per episode.
    size : int
        The size of the grid (default is 19x19).
    fully_observable : bool
        Whether the environment is fully observable.
    rewards : dict of str to float
        Stores the reward values for various actions.
    static_obj_map : list of list of int
        Stores a predefined static object map for level generation.
    actions : Actions
        Enum representing the possible actions the agent can take.
    action_space : gymnasium.spaces.Discrete
        The discrete action space for the environment.
    """

    REQUIRED_REWARD_KEYS: Set[str] = {
        "key_pickup",
        "correct_goal_reached",
        "incorrect_goal_reached",
    }
    VALID_OBJ_COLOR_INDEXES: List[int] = [0, 1, 3, 4, 5]  # Not including 2, which is gray
    GOAL_POS_MAP: Dict[Tuple[int, int], int] = {
        (9, 1): 1,
        (17, 9): 2,
        (9, 17): 3,
        (1, 9): 4,
    }
    MAX_STEPS: int = 200

    def __init__(
        self,
        rewards: Dict[str, float] = {
            "key_pickup": 0.2,
            "correct_goal_reached": 0.8,
            "incorrect_goal_reached": 0.4,
        },
        fully_observable: bool = True,
        **kwargs,
    ):

        mission_space: MissionSpace = MissionSpace(mission_func=self._gen_mission)
        self.size: int = 19
        self.fully_observable: bool = fully_observable
        self._validate_rewards(rewards)
        self.rewards: Dict[str, float] = rewards

        super().__init__(
            mission_space=mission_space,
            grid_size=self.size,
            see_through_walls=True,
            max_steps=self.MAX_STEPS,
            **kwargs,
        )

        self.actions: Type[Actions] = Actions
        self.action_space: spaces.Discrete = spaces.Discrete(len(self.actions))

    def _validate_rewards(self, rewards: Dict[str, float]) -> None:
        """
        Validates the rewards dictionary to ensure it has the correct keys,
        non-negative values, and proper sum relationships.

        Parameters
        ----------
        rewards : dict of str to float
            The rewards dictionary to validate. It must contain the exact keys:
            - "key_pickup"
            - "correct_goal_reached"
            - "incorrect_goal_reached"

        Raises
        ------
        ValueError
            If any required keys are missing or if extra keys are present.
        ValueError
            If any reward value is negative.
        ValueError
            If the sum of "key_pickup" and "correct_goal_reached" is not equal to 1.
        ValueError
            If "incorrect_goal_reached" is greater than or equal to "correct_goal_reached".
        """
        if set(rewards.keys()) != self.REQUIRED_REWARD_KEYS:
            raise ValueError(
                f"Invalid rewards dictionary. Expected keys: {self.REQUIRED_REWARD_KEYS}, "
                f"but got {set(rewards.keys())}"
            )
        for reward in rewards.values():
            if reward < 0:
                raise ValueError("No given reward can be negative")

        if rewards["key_pickup"] + rewards["correct_goal_reached"] != 1:
            raise ValueError(
                "The sum of rewards 'key_pickup' and 'correct_goal_reached' must equal 1"
            )

        if rewards["incorrect_goal_reached"] >= rewards["correct_goal_reached"]:
            raise ValueError(
                "The value of 'incorrect_goal_reached' must be smaller than the value of 'correct_goal_reached'"
            )

    @staticmethod
    def _gen_mission():
        return ""

    def _place_satic_objects(self) -> List[Tuple[int, int]]:
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

    def _place_key(self, empty_cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        key_pos: Tuple[int, int] = random.choice(empty_cells)
        self.grid.set(*key_pos, Key(COLOR_NAMES[self.chosen_color_i]))
        empty_cells.remove(key_pos)
        return empty_cells

    def _place_agent(self, empty_cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        self.agent_pos: Tuple[int, int] = random.choice(empty_cells)
        self.agent_dir: int = random.randint(0, 3)
        self.grid.set(*self.agent_pos, None)
        empty_cells.remove(self.agent_pos)
        return empty_cells

    def _place_boxes(self, empty_cells: List[Tuple[int, int]]) -> None:
        for color_i in self.VALID_OBJ_COLOR_INDEXES:
            boxes_n: int = random.randint(1, 4)
            if color_i == self.chosen_color_i:
                self.target_direction = (
                    boxes_n  # 1 for top, 2 for right, 3 for bottom and 4 for left
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
        self.grid.set(9, 0, NumberTile(color="blue"))
        self.grid.set(18, 9, NumberTile(color="red"))
        self.grid.set(9, 18, NumberTile(color="green"))
        self.grid.set(0, 9, NumberTile(color="yellow"))

    def _gen_grid(self, width, height):
        self.chosen_color_i: int = random.choice(
            self.VALID_OBJ_COLOR_INDEXES
        )  # Not including 2, which is gray
        self.grid: Grid = Grid(width, height)
        empty_cells: List[Tuple[int, int]] = self._place_satic_objects()
        self._place_boxes(empty_cells=empty_cells)
        empty_cells: List[Tuple[int, int]] = self._place_key(empty_cells=empty_cells)
        empty_cells: List[Tuple[int, int]] = self._place_agent(empty_cells=empty_cells)
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

        for i in range(self.agent_dir + 1):
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
                    reward = self.rewards["correct_goal_reached"]
                else:
                    reward = self.rewards["incorrect_goal_reached"]

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)
                    if fwd_cell.type == "key":
                        reward = self.rewards["key_pickup"]

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}
