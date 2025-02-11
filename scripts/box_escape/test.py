import itertools
import random
from collections import deque
from enum import IntEnum
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Box, Door, Goal, Key, Lava, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper


class Actions(IntEnum):
    left = 0
    right = 1
    forward = 2
    pickup = 3
    drop = 4
    toggle = 5


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
            "tab": Actions.pickup,
            "left shift": Actions.drop,
            "space": Actions.toggle,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)


class BoxEscape(MiniGridEnv):
    def __init__(
        self,
        multiple_keys=False,
        **kwargs,
    ):

        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.size = 19
        self.multiple_keys = multiple_keys

        max_steps = 200

        super().__init__(
            mission_space=mission_space,
            grid_size=self.size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

        self.actions = Actions
        self.action_space = spaces.Discrete(len(self.actions))
        self.wall_map = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 4, 4, 4, 4, 4, 4, 4, 1, 2, 1, 4, 4, 4, 4, 4, 4, 4, 1],
            [1, 4, 0, 0, 0, 0, 0, 4, 1, 3, 1, 4, 0, 0, 0, 0, 0, 4, 1],
            [1, 4, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 4, 1],
            [1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1],
            [1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1],
            [1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1],
            [1, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 1],
            [1, 1, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1, 1, 1],
            [1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 2, 1],
            [1, 1, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1, 1, 1],
            [1, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 1],
            [1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1],
            [1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1],
            [1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1],
            [1, 4, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 4, 1],
            [1, 4, 0, 0, 0, 0, 0, 4, 1, 3, 1, 4, 0, 0, 0, 0, 0, 4, 1],
            [1, 4, 4, 4, 4, 4, 4, 4, 1, 2, 1, 4, 4, 4, 4, 4, 4, 4, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]

    @staticmethod
    def _gen_mission():
        return ""

    def _get_random_coord(self):
        return (random.randint(1, 18), random.randint(1, 18))

    def place_boxes(self, color_i, empty_cells, first_box_pos, boxes_n):
        """Place multiple boxes in a connected configuration."""
        self.put_obj(Box(COLOR_NAMES[color_i]), *first_box_pos)
        empty_cells.remove(first_box_pos)

        placed_boxes = [first_box_pos]
        print(f"boxes n for color {COLOR_NAMES[color_i]}: {boxes_n}")
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

    def _place_objects(self) -> None:
        """Places objects including walls, doors, goal, and lava while ensuring connectivity."""
        empty_cells = []

        for y in range(self.size):
            for x in range(self.size):
                coord_value = self.wall_map[y][x]
                if coord_value == 1:
                    self.grid.set(x, y, Wall())
                elif coord_value == 2:
                    self.put_obj(Goal(), x, y)
                elif coord_value == 3:
                    door = Door(COLOR_NAMES[self.chosen_color_i], is_locked=True)
                    self.grid.set(x, y, door)
                elif coord_value == 0:
                    empty_cells.append((x, y))

        key_color_range = range(6) if self.multiple_keys else [self.chosen_color_i]
        for color_i in key_color_range:
            key_pos = random.choice(empty_cells)
            self.grid.set(*key_pos, Key(COLOR_NAMES[color_i]))
            empty_cells.remove(key_pos)

        # Place the agent at a random valid position
        self.agent_pos = random.choice(empty_cells)
        self.agent_dir = random.randint(0, 3)
        self.grid.set(*self.agent_pos, None)
        empty_cells.remove(self.agent_pos)

        for color_i in range(6):
            boxes_n = random.randint(1, 4)
            if color_i == self.chosen_color_i:
                self.target_direction = (
                    boxes_n  # 1 for top, 2 for right, 3 for bottom and 4 for left
                )
            first_box_pos = random.choice(empty_cells)
            self.place_boxes(
                empty_cells=empty_cells,
                color_i=color_i,
                first_box_pos=first_box_pos,
                boxes_n=boxes_n,
            )
        print(f"Chosen target_direction: {self.target_direction}")

    def _get_random_obj_coords(self) -> Tuple[int, int]:
        coords = random.choice(self.available_obj_coords)
        self.available_obj_coords.remove(coords)
        return coords

    def _gen_grid(self, width, height):
        self.chosen_color_i = random.randint(0, 5)
        self.grid = Grid(width, height)
        self._place_objects()
        self.mission = None


if __name__ == "__main__":
    env = BoxEscape(multiple_keys=False)
    env = ImgObsWrapper(RGBImgObsWrapper(env))
    obs, _ = env.reset()
    plt.imshow(X=obs)
    plt.show
    """ manual_control = CustomManualControl(env, seed=42)
    manual_control.start() """
