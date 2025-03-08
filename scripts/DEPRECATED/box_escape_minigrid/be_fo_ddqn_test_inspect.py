import argparse
from typing import cast

import box_escape
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from keras.api.models import Sequential
from keras.api.saving import load_model
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
from tensorflow.keras.models import Model

from agents import DDQNAgent


def visualize_activations(activations: np.ndarray, layer_name: str):
    """
    Visualize the activations of a single convolutional layer.

    Parameters
    ----------
    activations : np.ndarray
        The activations output from a convolutional layer, expected shape (batch, height, width, channels).
    layer_name : str
        The name of the layer being visualized.
    """
    if len(activations.shape) != 4:
        raise ValueError("Expected activations with 4 dimensions (batch, height, width, channels)")

    activations = activations[0]  # Remove batch dimension
    num_features = activations.shape[-1]  # Number of feature maps

    cols = min(8, num_features)  # Limit to 8 columns per row
    rows = (num_features // cols) + (1 if num_features % cols else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    fig.suptitle(f"Activations of Layer: {layer_name}", fontsize=16)

    for i in range(num_features):
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
        ax.imshow(activations[:, :, i])
        ax.axis("off")

    # Hide any unused subplots
    for j in range(num_features, rows * cols):
        fig.delaxes(axes.flatten()[j])

    plt.show()


def _create_activation_model(model: Sequential, layer_index: int) -> Model:
    """
    Create an intermediate model to extract activations from a specific layer.

    Parameters
    ----------
    layer_index : int
        The index of the layer in the Sequential model.

    Returns
    -------
    Model
        A Keras Model that outputs the activations of the specified layer.
    """
    return Model(
        inputs=model.layers[0].input,
        outputs=model.layers[layer_index].output,  # type: ignore
    )


if __name__ == "__main__":

    model_path = "models/BoxEscape/gentle-dew-101/1739424389_model____0.0832avg____0.9960max___-0.8160min.keras"

    env = gym.make(
        id="BoxEscape-v1",
        render_mode="human",
        fully_observable=True,
        curriculum_level=1,
    )
    env = ImgObsWrapper(RGBImgObsWrapper(env))
    agent = DDQNAgent(env=env, obervation_normalization_type="image")

    model = load_model(filepath=model_path)
    model = cast(Sequential, model)

    LAYER_INDEXES = [i for i in range(len(model.layers))]

    obs, _ = env.reset()
    obs = np.expand_dims(obs, axis=0)  # Shape becomes (1, 120, 120, 3)

    # Get the names of convolutional layers
    for LAYER_INDEX in LAYER_INDEXES:
        activation_model = _create_activation_model(model=model, layer_index=LAYER_INDEX)
        activations = activation_model.predict(obs / 255)
        conv_layer_name = model.layers[LAYER_INDEX].name
        # Visualize
        visualize_activations(activations, conv_layer_name)

    # agent.test(model=model, episodes=1)
