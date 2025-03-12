import numpy as np
import tensorflow as tf
from keras.api.layers import Input
from keras.api.models import Model, Sequential
from tensorflow import Tensor

from .cav import Cav
from .model_activation_obtainer import ModelActivationObtainer


class TcavScorer:
    """
    A class for computing TCAV (Testing with Concept Activation Vectors) scores for Keras Sequential DNN models.

    This class provides methods to create output models from a given layer, calculate gradients of the model's output
    with respect to layer activations for a specified target class, and compute the TCAV score based on the directional
    derivatives obtained from these gradients and a provided Concept Activation Vector (CAV).
    """

    def _create_output_model(self, layer_index: int, model: Sequential) -> Model | None:
        """
        Create an output model consisting of the layers above a specified layer index.

        This method constructs a new Keras model that takes as input the activations from the specified layer and
        applies all subsequent layers of the original model to produce the final output. If the specified layer is the
        last layer of the model, no output model is created and None is returned.

        Parameters
        ----------
        layer_index : int
            The index of the layer from which the output model should be constructed.
        model : Sequential
            The Keras sequential model of interest.

        Returns
        -------
        Model or None
            A new Keras model representing the layers above the specified layer index, or None if the specified layer
            is the last layer in the model.
        """
        # Defining output model, consisting of every layer above layer_index, if any
        if layer_index < len(model.layers) - 1:
            new_input = Input(shape=model.layers[layer_index].output.shape[1:])  # type: ignore

            x = new_input
            for layer in model.layers[layer_index + 1 :]:
                x = layer(x)

            return Model(inputs=new_input, outputs=x)

        else:
            # If layer_index equals the last layer index of the original model, the activations of the activation model are the outputs
            return None

    def _calculate_gradients(
        self,
        model_activation_obtainer: ModelActivationObtainer,
        model: Sequential,
        layer_index: int,
        observations: np.ndarray,
        target_class: int,
    ):
        """
        Calculate the gradients of the model's output with respect to the activations of a specified layer.

        This method obtains the activations for the specified layer using the provided activation obtainer, constructs an
        output model for the subsequent layers (if any), and computes the gradient of the target class's output with respect
        to these activations. The gradients are then flattened and returned as a NumPy array.

        Parameters
        ----------
        model_activation_obtainer : ModelActivationObtainer
            An instance used to obtain the activations from a specific layer of the model.
        model : Sequential
            The original Keras sequential model.
        layer_index : int
            The index of the layer at which to obtain activations.
        observations : np.ndarray
            The input data for which the activations are computed.
        target_class : int
            The index of the target class for which the gradient is calculated.

        Returns
        -------
        np.ndarray
            A 2D NumPy array containing the flattened gradients for each observation.
        """
        # Defining activation model, consisting of every layer of the original model up until layer_index, inclusive

        activations = model_activation_obtainer.get_layer_activations(
            layer_index=layer_index, model_inputs=observations, flatten=False
        )

        output_model: Model | None = self._create_output_model(layer_index=layer_index, model=model)

        activations_tensor: Tensor = tf.convert_to_tensor(activations)

        with tf.GradientTape() as tape:
            tape.watch(activations_tensor)
            output: Model | Tensor = (
                output_model(activations_tensor, training=False)
                if output_model
                else activations_tensor
            )
            class_output = output[:, target_class]  # type: ignore

        gradient = tape.gradient(target=class_output, sources=activations_tensor)

        gradients_flat = tf.reshape(gradient, (gradient.shape[0], -1)).numpy()  # type: ignore

        return gradients_flat

    def calculate_tcav_score(
        self,
        model: Sequential,
        model_activation_obtainer: ModelActivationObtainer,
        layer_index: int,
        cav: Cav,
        target_class: int,
        target_class_examples: np.ndarray,
    ):
        """
        Calculate the TCAV score for a given target class.

        This method computes the gradients for the specified layer and target class, calculates the directional derivatives
        by taking the dot product of the gradients and the provided Concept Activation Vector (CAV), and then computes the
        TCAV score as the mean fraction of directional derivatives that are positive.

        Parameters
        ----------
        model : Sequential
            The Keras sequential model of interest.
        model_activation_obtainer : ModelActivationObtainer
            An instance used to obtain activations from a specific layer of the model.
        layer_index : int
            The index of the layer from which to compute activations.
        cav : Cav
            An object containing the Concept Activation Vector.
        target_class : int
            The index of the target class for which the TCAV score is computed.
        target_class_examples : np.ndarray
            Input examples corresponding to the target class.

        Returns
        -------
        float
            The computed TCAV score representing the fraction of positive directional derivatives.
        """

        gradients = self._calculate_gradients(
            model_activation_obtainer=model_activation_obtainer,
            model=model,
            layer_index=layer_index,
            target_class=target_class,
            observations=target_class_examples,
        )

        directional_derivatives = np.dot(gradients, cav.vector)

        tcav_score = np.mean(directional_derivatives > 0)

        return tcav_score
