from enum import IntEnum


class Actions(IntEnum):
    """
    The possible actions in the BoxEscape environment.

    Actions correspond to discrete movements and interactions that the agent
    can perform within the environment:

    - `left (0)`: Rotate the agent 90 degrees to the left.
    - `right (1)`: Rotate the agent 90 degrees to the right.
    - `forward (2)`: Move the agent one step forward in the direction it is facing.
    - `pickup (3)`: Pick up an object (such as a key or a box) if the agent is facing it.
    - `drop (4)`: Drop the currently held object at the agent's current position.
    - `toggle (5)`: Toggle/interact with an object, such as opening a door or activating a switch.

    These actions allow the agent to navigate and interact with the environment
    to solve puzzles, unlock paths, and reach the goal.
    """

    left = 0
    right = 1
    forward = 2
    pickup = 3
    drop = 4
    toggle = 5
