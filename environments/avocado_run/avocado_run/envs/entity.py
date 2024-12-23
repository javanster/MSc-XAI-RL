import random
from typing import Optional, Tuple

import numpy as np


class Entity:
    """
    A class for an entity used in the AvocadoRun environment
    """

    def __init__(
        self, grid_side_length: int, starting_position: Optional[Tuple[int, int]] = None
    ) -> None:
        if starting_position and any(
            starting_coord >= grid_side_length or starting_coord < 0
            for starting_coord in starting_position
        ):
            raise ValueError(
                "A starting coordinate may not be equal to or exceed grid_side_length, or be below 0"
            )
        if starting_position:
            self.x = starting_position[0]
            self.y = starting_position[1]
        else:
            self.x = np.random.randint(0, grid_side_length)
            self.y = np.random.randint(0, grid_side_length)
        self.grid_side_length = grid_side_length

    def __sub__(self, other: "Entity") -> Tuple[int, int]:
        return (self.x - other.x, self.y - other.y)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False
        return self.x == other.x and self.y == other.y

    def action(self, choice) -> None:
        if choice == 0:
            self.move(y=-1)  # Up
        elif choice == 1:
            self.move(x=1)  # Right
        elif choice == 2:
            self.move(y=1)  # Down
        elif choice == 3:
            self.move(x=-1)  # Left
        elif choice == 4:
            self.move()  # Do nothing

    def move(self, x=0, y=0) -> None:
        self.x = max(0, min(self.x + x, self.grid_side_length - 1))
        self.y = max(0, min(self.y + y, self.grid_side_length - 1))

    def move_towards_target(self, other) -> None:
        potential_directions = []

        if self.x < other.x:
            potential_directions.append(("x", 1))  # Move right
        elif self.x > other.x:
            potential_directions.append(("x", -1))  # Move left

        if self.y < other.y:
            potential_directions.append(("y", 1))  # Move down
        elif self.y > other.y:
            potential_directions.append(("y", -1))  # Move up

        if potential_directions:
            axis, delta = random.choice(potential_directions)
            if axis == "x":
                self.move(x=delta)
            else:
                self.move(y=delta)

    def random_action(self) -> None:
        # Generate possible moves to avoid "do_nothing"
        possible_moves = []
        if self.y > 0:
            possible_moves.append(0)  # Up
        if self.x < self.grid_side_length - 1:
            possible_moves.append(1)  # Right
        if self.y < self.grid_side_length - 1:
            possible_moves.append(2)  # Down
        if self.x > 0:
            possible_moves.append(3)  # Left

        rand_act = random.choice(possible_moves)
        self.action(rand_act)
