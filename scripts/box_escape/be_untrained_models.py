import gem_collector
import gymnasium as gym
from keras.api.layers import Conv2D, Dense, Flatten, Input
from keras.api.models import Sequential
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper, RGBImgPartialObsWrapper


def get_be_untrained_conv_ff_model():
    """
    Creates and returns an untrained and uncompiled feed-forward convolutional neural network (CNN) model
    for the FULLY OBSERVED VERSION of the BoxEscape environment.

    Returns
    -------
    Sequential
        A Keras Sequential model with convolutional and dense layers, designed
        for processing visual observations from the FULLY OBSERVED of the BoxEscape environment.
    """
    env = gym.make(
        id="BoxEscape-v1",
        render_mode=None,
        fully_observable=True,
    )
    env = ImgObsWrapper(RGBImgObsWrapper(env))
    obs, _ = env.reset()

    input_shape = obs.shape
    output_shape = env.action_space.n

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, kernel_size=8, strides=4, activation="relu"))
    model.add(Conv2D(64, kernel_size=4, strides=2, activation="relu"))
    model.add(Conv2D(64, kernel_size=3, strides=1, activation="relu"))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(units=output_shape, activation="linear"))

    return model
