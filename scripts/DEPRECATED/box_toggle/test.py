import itertools
import random
from enum import IntEnum
from typing import Any, SupportsFloat, Tuple

from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Box, Door, Goal, Key, Lava, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper


class Actions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2
    toggle = 3


class CustomManualControl(ManualControl):
    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "space": Actions.toggle,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)


class BoxToggle(MiniGridEnv):
    def __init__(
        self,
        size=16,
        **kwargs,
    ):

        mission_space = MissionSpace(mission_func=self._gen_mission)

        max_steps = 200

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

        self.actions = Actions
        self.action_space = spaces.Discrete(len(self.actions))

    @staticmethod
    def _gen_mission():
        return ""

    def _gen_walls(self, width, height):
        for i in range(height):
            self.grid.set(1, i, Wall())
            self.grid.set(width - 1, i, Wall())
        for i in range(width):
            self.grid.set(i, 0, Wall())
            self.grid.set(i, height - 2, Wall())

        for i in [1, 2, height - 4, height - 3, height - 1]:
            self.grid.set(0, i, Wall())
        for i in [2, width - 3, width - 2]:
            self.grid.set(i, height - 1, Wall())

    def _gen_boxes(self, width, height):
        self.n_target_blue_boxes = random.randint(0, 10)
        n_target_red_boxes = 10 - self.n_target_blue_boxes
        n_blue_boxes = random.choice([n for n in range(0, 11) if n != self.n_target_blue_boxes])
        n_red_boxes = 10 - n_blue_boxes

        for _ in range(n_blue_boxes):
            coord = self._get_random_obj_coords()
            self.put_obj(Box(COLOR_NAMES[0]), coord[0], coord[1])
        for _ in range(n_red_boxes):
            coord = self._get_random_obj_coords()
            self.put_obj(Box(COLOR_NAMES[4]), coord[0], coord[1])

        for i in range(self.n_target_blue_boxes):
            self.put_obj(Box(COLOR_NAMES[0]), i + 3, height - 1)

        for i in range(n_target_red_boxes):
            self.put_obj(Box(COLOR_NAMES[4]), 0, height - 4 - i)

    def _get_random_obj_coords(self) -> Tuple[int, int]:
        coords = random.choice(self.available_obj_coords)
        self.available_obj_coords.remove(coords)
        return coords

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # The offsets ensure that no placed obj is ever placed right next to one another, so that
        # the agent can get to every one. The randomness is there to vary the grid
        x_offset = random.randint(0, 1)
        y_offset = random.randint(0, 1)

        self.available_obj_coords = [
            (x, y)
            for (x, y) in itertools.product(
                [x for x in range(2 + x_offset, width - 1, 2)],
                [y for y in range(1 + y_offset, height - 3, 2)],
            )
        ]

        # Place a goal square
        goal_pos = self._get_random_obj_coords()
        self.put_obj(Goal(), goal_pos[0], goal_pos[1])

        self._gen_walls(width=width, height=height)
        self._gen_boxes(width=width, height=height)

        # Place the agent
        self.agent_pos = self._get_random_obj_coords()
        self.agent_dir = random.randint(0, 3)
        self.grid.set(*self.agent_pos, None)

        self.mission = None

    def _current_blue_box_n(self) -> int:
        blue_box_count = 0
        for x in range(2, self.grid.width):
            for y in range(self.grid.height - 2):
                obj = self.grid.get(x, y)
                if isinstance(obj, Box) and obj.color == "blue":
                    blue_box_count += 1
        return blue_box_count

    def _is_done(self) -> bool:
        blue_box_count = self._current_blue_box_n()
        return blue_box_count == self.n_target_blue_boxes

    def _get_toggle_reward(self, box_color: str) -> float:
        current = self._current_blue_box_n()
        target = self.n_target_blue_boxes

        if box_color == "red":
            return 0.0024 if current < target else -0.0025
        elif box_color == "blue":
            return 0.0024 if current > target else -0.0025
        else:
            raise ValueError(f"Invalid box_color: {box_color}")

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
                if self._is_done():
                    terminated = True
                    reward = 0.976

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell and isinstance(fwd_cell, Box):
                original_color = fwd_cell.color
                reward = self._get_toggle_reward(original_color)
                if original_color == "blue":
                    fwd_cell.color = "red"
                elif original_color == "red":
                    fwd_cell.color = "blue"

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}


if __name__ == "__main__":
    env = BoxToggle(render_mode="human")
    env = ImgObsWrapper(RGBImgObsWrapper(env))
    manual_control = CustomManualControl(env, seed=42)
    manual_control.start()
