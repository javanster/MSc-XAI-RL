from minigrid.core.world_object import WorldObj
from minigrid.utils.rendering import fill_coords, point_in_rect


class NumberTile(WorldObj):
    """
    A tile displaying a number, represented by different colors.

    This class extends `WorldObj` and renders a number on a tile
    based on the assigned color. The number is drawn using
    simple geometric shapes.

    Parameters
    ----------
    color : str, optional
        The color of the tile, which determines the number displayed.


    Attributes
    ----------
    color : str
        The color of the tile, used to determine the number representation.
    """

    def __init__(self, color="blue"):
        super().__init__("ball", color)  # Using ball to not trigger MiniGrid error

    def can_pickup(self):
        """
        Determines whether the tile can be picked up.

        Returns
        -------
        bool
            Always returns False, as NumberTile objects cannot be picked up.
        """
        return False

    def render(self, img):
        """
        Renders the tile with a number, based on its color.

        Parameters
        ----------
        img : numpy.ndarray
            The image array where the tile will be drawn.
        """
        white = (255, 255, 255)

        if self.color == "blue":  # 1
            fill_coords(img, point_in_rect(0.35, 0.65, 0.75, 0.85), white)
            fill_coords(img, point_in_rect(0.45, 0.55, 0.25, 0.75), white)
            fill_coords(img, point_in_rect(0.35, 0.55, 0.15, 0.25), white)

        elif self.color == "red":  # 2
            fill_coords(img, point_in_rect(0.25, 0.85, 0.75, 0.85), white)
            fill_coords(img, point_in_rect(0.45, 0.55, 0.65, 0.75), white)
            fill_coords(img, point_in_rect(0.55, 0.65, 0.55, 0.65), white)
            fill_coords(img, point_in_rect(0.65, 0.75, 0.45, 0.55), white)
            fill_coords(img, point_in_rect(0.75, 0.85, 0.25, 0.45), white)
            fill_coords(img, point_in_rect(0.35, 0.75, 0.15, 0.25), white)
            fill_coords(img, point_in_rect(0.25, 0.35, 0.25, 0.35), white)

        elif self.color == "green":  # 3
            fill_coords(img, point_in_rect(0.25, 0.75, 0.75, 0.85), white)
            fill_coords(img, point_in_rect(0.75, 0.85, 0.55, 0.75), white)
            fill_coords(img, point_in_rect(0.45, 0.75, 0.45, 0.55), white)
            fill_coords(img, point_in_rect(0.75, 0.85, 0.25, 0.45), white)
            fill_coords(img, point_in_rect(0.25, 0.75, 0.15, 0.25), white)

        elif self.color == "yellow":  # 4
            fill_coords(img, point_in_rect(0.65, 0.75, 0.55, 0.85), white)
            fill_coords(img, point_in_rect(0.35, 0.65, 0.45, 0.55), white)
            fill_coords(img, point_in_rect(0.25, 0.35, 0.25, 0.45), white)
            fill_coords(img, point_in_rect(0.65, 0.75, 0.25, 0.45), white)
