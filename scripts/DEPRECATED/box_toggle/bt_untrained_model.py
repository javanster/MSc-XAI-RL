from keras.api.layers import Conv2D, Dense, Flatten, Input
from keras.api.models import Sequential
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper

from scripts.box_toggle.test import BoxToggle


def get_bt_untrained_model():
    """
    Creates and returns an untrained and uncompiled convolutional neural network (CNN) model
    for the GemCollector environment.

    Returns
    -------
    Sequential
        A Keras Sequential model with convolutional and dense layers, designed
        for processing visual observations from the GemCollector environment.
    """
    env = BoxToggle()
    env = ImgObsWrapper(RGBImgObsWrapper(env))

    input_shape = env.observation_space.shape
    output_shape = env.action_space.n

    model: Sequential = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, kernel_size=3, activation="relu", padding="same"))
    model.add(Conv2D(64, kernel_size=3, activation="relu", padding="same"))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(units=output_shape, activation="linear"))

    return model
