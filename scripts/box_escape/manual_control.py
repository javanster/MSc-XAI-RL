from minigrid.manual_control import ManualControl

from .actions import Actions


class CustomManualControl(ManualControl):
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
            "left shift": Actions.drop,
            "space": Actions.toggle,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)
