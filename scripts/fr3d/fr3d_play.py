import gymnasium
from miniworld.manual_control import ManualControl

if __name__ == "__main__":
    env = gymnasium.make("MiniWorld-FourRooms-v0", render_mode="human")

    manual_control = ManualControl(env, True, True)
    manual_control.run()
