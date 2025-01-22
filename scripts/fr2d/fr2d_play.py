import gymnasium
from minigrid.manual_control import ManualControl

if __name__ == "__main__":
    env = gymnasium.make("MiniGrid-FourRooms-v0", render_mode="human")

    manual_control = ManualControl(env)
    manual_control.start()
