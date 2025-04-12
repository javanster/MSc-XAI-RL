from typing import List, Tuple

import numpy as np
from keras.api.layers import Dense, Input
from keras.api.models import Sequential
from keras.api.optimizers import Adam

from utils import ModelActivationObtainer

from .ccm import CCM


class CCM_NN(CCM):
    """
    Concept Completeness Model using a Neural Network decoder.

    This subclass of CCM uses a feedforward neural network (Sequential API)
    to learn the mapping from concept scores to model outputs. It can operate
    in classification mode (predicting actions) or regression mode (predicting
    Q-values) depending on the `all_q` flag.

    Parameters
    ----------
    model_activation_obtainer : ModelActivationObtainer
        Utility to extract intermediate layer activations from the model to be explained.
    num_classes : int
        Number of output classes for classification tasks.
    X_train : np.ndarray
        Training input data (observations).
    X_val : np.ndarray
        Validation input data (observations).
    Y_train : np.ndarray
        Training labels (action indices or Q-values).
    Y_val : np.ndarray
        Validation labels (action indices or Q-values).
    all_q : bool, optional
        If True, use regression to predict full Q-values; if False, use classification. Default is False.
    """

    def __init__(
        self,
        model_activation_obtainer: ModelActivationObtainer,
        num_classes: int,
        X_train: np.ndarray,
        X_val: np.ndarray,
        Y_train: np.ndarray,
        Y_val: np.ndarray,
        all_q: bool = False,
    ) -> None:
        super().__init__(
            model_activation_obtainer=model_activation_obtainer,
            num_classes=num_classes,
            X_train=X_train,
            X_val=X_val,
            Y_train=Y_train,
            Y_val=Y_val,
            all_q=all_q,
        )

    def _train_ccm_model(
        self,
        concept_scores: List[np.ndarray],
    ) -> Sequential:
        """
        Train a neural network decoder on concept scores to predict model outputs.

        Parameters
        ----------
        concept_scores : List[np.ndarray]
            List of concept score vectors for the training data.

        Returns
        -------
        Sequential
            A trained Keras Sequential model that maps concept scores to outputs.
            - Uses softmax activation and categorical cross-entropy loss in classification mode.
            - Uses linear activation and mean squared error loss in regression mode.
        """
        X = np.vstack(concept_scores)
        input_dim = X.shape[1]

        if self.all_q:
            output_dim = self.Y_train.shape[1]
            loss = "mse"
            final_activation = "linear"
        else:
            output_dim = self.num_classes
            loss = "categorical_crossentropy"
            final_activation = "softmax"

        model_nn = Sequential()
        model_nn.add(Input(shape=(input_dim,)))
        model_nn.add(Dense(64, activation="relu"))
        model_nn.add(Dense(output_dim, activation=final_activation))

        model_nn.compile(optimizer=Adam(learning_rate=0.001), loss=loss)  # type: ignore

        if self.all_q:
            model_nn.fit(X, self.Y_train, epochs=100, batch_size=32)
        else:
            model_nn.fit(X, self.Y_train, epochs=100, batch_size=32)

        return model_nn
