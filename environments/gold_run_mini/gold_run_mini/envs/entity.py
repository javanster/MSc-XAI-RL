from typing import Optional, Tuple

import numpy as np


class Entity:
    """
    A class for an entity used in the BoxEscapeLite environment
    """

    def __init__(
        self,
        grid_side_length: int,
        color: Tuple[int, int, int],
        starting_position: Optional[Tuple[int, int]] = None,
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
        self.color = color

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
            pass  # Do nothing

    def move(self, x=0, y=0) -> None:
        self.x = max(0, min(self.x + x, self.grid_side_length - 1))
        self.y = max(0, min(self.y + y, self.grid_side_length - 1))
