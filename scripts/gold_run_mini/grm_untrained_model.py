from keras.api.layers import Conv2D, Dense, Flatten, Input
from keras.api.models import Sequential


def get_grm_untrained_model(input_shape, output_shape):
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
