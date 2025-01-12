import numpy as np
from keras import Model
from keras.api.models import Sequential


class ModelActivationObtainer:
    """
    A utility class for obtaining activations from layers of a Keras sequential model.

    This class creates intermediate models to fetch the activations from each layer
    of a given Keras Sequential model for provided inputs. It also flattens activations
    for Conv2D layers.

    Parameters
    ----------
    model : Sequential
        The Keras Sequential model whose layer activations will be extracted.
    """

    def __init__(self, model: Sequential) -> None:
        self.model: Sequential = model
        self._activation_models = {
            layer_index: self._create_activation_model(layer_index=layer_index)
            for layer_index in range(len(model.layers))
        }

    def _create_activation_model(self, layer_index: int) -> Model:
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
            inputs=self.model.layers[0].input,
            outputs=self.model.layers[layer_index].output,  # type: ignore
        )

    def get_layer_activations(self, layer_index: int, model_inputs: np.ndarray) -> np.ndarray:
        """
        Get the activations from a specific layer for the given inputs.

        This method uses the intermediate model for the specified layer to compute
        the activations. If the layer is a Conv2D layer, the activations are flattened
        into a 2D array.

        Parameters
        ----------
        layer_index : int
            The index of the layer whose activations are to be fetched.
        model_inputs : np.ndarray
            The inputs to the model for which activations are computed.
            The shape of this array should match the model's input shape.

        Returns
        -------
        np.ndarray
            The activations from the specified layer. If the layer is a Conv2D layer,
            the activations are flattened to a shape of `(batch_size, flattened_activations)`.

        Notes
        -----
        Conv2D layer activations are reshaped from `(batch_size, height, width, channels)`
        to `(batch_size, height * width * channels)`.
        """
        activation_model: Model = self._activation_models[layer_index]

        activations: np.ndarray = activation_model.predict(model_inputs)

        # If the layer is a Conv2D layer, flatten the output
        if len(activations.shape) == 4:
            batch_size, height, width, channels = activations.shape
            activations = activations.reshape(batch_size, height * width * channels)

        return activations
