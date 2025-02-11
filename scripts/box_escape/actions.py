from enum import IntEnum


class Actions(IntEnum):
    """
    The possible actions in the BoxEscape environment.

    These actions define the discrete movement and interaction capabilities of
    the agent as it navigates the environment.

    Attributes
    ----------
    left : int
        Rotate the agent 90 degrees to the left.
    right : int
        Rotate the agent 90 degrees to the right.
    forward : int
        Move the agent one step forward in the direction it is facing.
    pickup : int
        Pick up an object (such as a key or a box) if the agent is facing it.
    drop : int
        Drop the currently held object at the agent's current position.
    toggle : int
        Toggle or interact with an object, such as opening a door or activating a switch.
    """

    left = 0
    right = 1
    forward = 2
    pickup = 3
    drop = 4
    toggle = 5
