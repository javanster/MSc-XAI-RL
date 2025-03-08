import box_escape
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper

from agents import BoxEscapeHumanPlayer

if __name__ == "__main__":
    env = gym.make(
        id="BoxEscape-v1",
        render_mode="human",
        fully_observable=True,
        curriculum_level=2,
    )
    player = BoxEscapeHumanPlayer(env)
    player.start()
