from enum import IntEnum

from minigrid.manual_control import ManualControl


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
    toggle : int
        Toggle or interact with an object, such as opening a door or activating a switch.
    """

    left = 0
    right = 1
    forward = 2
    pickup = 3
    toggle = 4


class BoxEscapeHumanPlayer(ManualControl):
    """
    A custom manual control interface for interacting with the BoxEscape environment.

    This class extends the standard ManualControl class and maps keyboard inputs
    to predefined actions in the environment.
    """

    def key_handler(self, event):
        """
        Handles keyboard inputs and translates them into environment actions.

        Parameters
        ----------
        event : pygame.event.Event
            The keyboard event triggered by the user.
        """
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
            "space": Actions.toggle,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)

    def step(self, action: Actions):
        _, reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.unwrapped.step_count}, reward={reward}")

        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.env.render()
