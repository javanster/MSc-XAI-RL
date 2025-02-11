import box_escape
import gymnasium as gym
import matplotlib.pyplot as plt
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper, RGBImgPartialObsWrapper

if __name__ == "__main__":
    env_po = gym.make(
        id="BoxEscape-v1",
        render_mode=None,
        fully_observable=False,
    )
    env_po = ImgObsWrapper(RGBImgPartialObsWrapper(env_po))
    obs, _ = env_po.reset()
    plt.imshow(X=obs)
    plt.title("Partially observable obs")
    plt.show()

    env_fo = gym.make(
        id="BoxEscape-v1",
        render_mode=None,
        fully_observable=True,
    )
    env_fo = ImgObsWrapper(RGBImgObsWrapper(env_fo))
    obs, _ = env_fo.reset()
    plt.imshow(X=obs)
    plt.title("Fully observable obs")
    plt.show()

    plt.close()
