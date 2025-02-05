from typing import Dict, Tuple

from .entity import Entity


class Miner(Entity):
    """
    A class representing a Miner, a specialized type of Entity in the grid-based environment.

    Attributes
    ----------
    is_moving_left : bool
        Indicates whether the Miner is currently moving left.
    obj_drop_x_coordinates : Dict[int, str]
        A dictionary mapping x-coordinates to object types dropped by the Miner.
    grid_side_length : int
        The size of one side of the square grid environment (inherited from Entity).
    color : Tuple[int, int, int]
        The RGB color of the Miner (inherited from Entity).
    x : int
        The x-coordinate of the Miner in the grid (inherited from Entity).
    y : int
        The y-coordinate of the Miner in the grid (inherited from Entity).
    """

    def __init__(
        self,
        grid_side_length: int,
        color: Tuple[int, int, int],
        is_moving_left: bool,
        starting_position: Tuple[int, int] | None = None,
    ) -> None:
        super().__init__(
            grid_side_length=grid_side_length,
            color=color,
            starting_position=starting_position,
        )

        self.is_moving_left = is_moving_left
        self.obj_drop_x_coordinates: Dict[int, str] = {}
