from typing import Dict, Optional, Tuple

import numpy as np


class Entity:
    """
    A class representing an entity in the GemCollector environment.

    Attributes
    ----------
    x : int
        The x-coordinate of the entity in the grid.
    y : int
        The y-coordinate of the entity in the grid.
    grid_side_length : int
        The size of one side of the square grid environment.
    color : Tuple[int, int, int]
        The RGB color of the entity.
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
        """
        Compare two entities for equality based on their positions.

        Parameters
        ----------
        other : object
            The object to compare against.

        Returns
        -------
        bool
            True if the other object is an Entity and has the same (x, y) coordinates,
            otherwise False.

        Raises
        ------
        ValueError
            If the other object is not an instance of Entity.
        """
        if not isinstance(other, Entity):
            raise ValueError("Comparison object must be an instance of Entity.")
        return self.x == other.x and self.y == other.y

    def action(self, choice: int) -> None:
        """
        Perform an action based on the given choice.

        Parameters
        ----------
        choice : int
            The action choice, where:
            - 0 moves left
            - 1 moves right
            - 2 does nothing
            - 3 moves down
        """
        if choice == 0:
            self._move(x=-1)  # Left
        elif choice == 1:
            self._move(x=1)  # Right
        elif choice == 2:
            self._move()  # Do nothing
        elif choice == 3:
            self._move(y=1)  # Down

    def _move(self, x=0, y=0) -> None:
        """
        Move the entity within the grid boundaries.

        Parameters
        ----------
        x : int, optional
            The movement along the x-axis (default is 0).
        y : int, optional
            The movement along the y-axis (default is 0).
        """
        self.x = max(0, min(self.x + x, self.grid_side_length - 1))
        self.y = max(0, min(self.y + y, self.grid_side_length - 1))
