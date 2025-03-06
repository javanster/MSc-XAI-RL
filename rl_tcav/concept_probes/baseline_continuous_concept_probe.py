from typing import List, cast

import numpy as np
from keras.api.layers import Dense, Input
from keras.api.models import Sequential
from keras.api.optimizers import SGD
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ..cav import Cav
from ..concept_classes.continuous_concept import ContinuousConcept
from ..model_activation_obtainer import ModelActivationObtainer
from .concept_probe import ConceptProbe


class BaselineContinuousConceptProbe(ConceptProbe):
    """
    A simple linear regression probe for continuous concepts.

    This probe learns a linear mapping from activations to continuous concept labels
    using a single-layer regression model. The probe score is computed using both
    the coefficient of determination (R²) and Mean Squared Error (MSE).

    Parameters
    ----------
    concept : ContinuousConcept
        The continuous concept to probe.
    model_activation_obtainer : ModelActivationObtainer
        The activation obtainer used to extract activations from the model.
    model_layer_index : int
        The index of the layer from which to obtain activations.

    Attributes
    ----------
    concept : ContinuousConcept
        The continuous concept associated with this probe.
    linear_regressor : Sequential or None
        The linear regression model used for probing. None if the probe has not been trained.
    model_activation_obtainer : ModelActivationObtainer
        The activation obtainer used to fetch layer activations.
    model_layer_index : int
        The index of the model layer used for activations.
    concept_probe_score : float or None
        The probe score computed as the coefficient of determination (R²). None if the probe has not been trained.
    concept_probe_mse : float or None
        The probe's Mean Squared Error (MSE) on the training-validation split. None if the probe has not been trained.
    concept_probe_score_on_validation_set : float or None
        The probe's R² score when evaluated on an external validation dataset. None if not validated.
    concept_probe_mse_on_validation_set : float or None
        The probe's MSE when evaluated on an external validation dataset. None if not validated.
    """

    def __init__(
        self,
        concept: ContinuousConcept,
        model_activation_obtainer: ModelActivationObtainer,
        model_layer_index: int,
    ) -> None:
        self.concept: ContinuousConcept = concept
        self.linear_regressor = None
        self.model_activation_obtainer = model_activation_obtainer
        self.model_layer_index: int = model_layer_index
        self.concept_probe_score: float | None = None
        self.concept_probe_mse: float | None = None
        self.concept_probe_score_on_validation_set: float | None = None
        self.concept_probe_mse_on_validation_set: float | None = None

    def train_concept_probe(self) -> None:
        """
        Trains the concept probe using a simple linear regression model.

        The activations from the specified model layer are extracted and used to fit
        a regression model predicting the continuous concept values. The model's
        performance is evaluated using both R² and Mean Squared Error (MSE).

        Raises
        ------
        ValueError
            If there are fewer than 10 concept examples.
        ValueError
            If the number of examples and labels are not equal.
        """

        if len(self.concept.examples) < 10:
            raise ValueError(
                "There needs to be a minimum of 10 concept examples in order to train a concept probe"
            )

        if len(self.concept.examples) != len(self.concept.labels):
            raise ValueError(
                "The number of examples and labels in the given continuous concept sre not equal"
            )

        example_layer_activations = self.model_activation_obtainer.get_layer_activations(
            layer_index=self.model_layer_index,
            model_inputs=np.array(self.concept.examples),
            flatten=True,
        )

        X = example_layer_activations
        y = np.array(self.concept.labels)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        input_dim = X_train.shape[1]

        self.linear_regressor = Sequential()
        self.linear_regressor.add(Input(shape=(input_dim,)))
        self.linear_regressor.add(Dense(1, activation="linear"))

        self.linear_regressor.compile(
            optimizer=SGD(learning_rate=0.01),  # type: ignore
            loss="mean_squared_error",
            metrics=["mse"],
        )

        self.linear_regressor.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)  # type: ignore

        y_pred = self.linear_regressor.predict(X_val).flatten()
        self.concept_probe_score = cast(float, r2_score(y_val, y_pred))
        self.concept_probe_mse = cast(float, mean_squared_error(y_val, y_pred))

    def extract_cav(self) -> Cav:
        """
        Extracts the Concept Activation Vector (CAV) from the trained probe.

        The CAV is derived from the learned weights of the linear regression model.

        Returns
        -------
        Cav
            The Concept Activation Vector containing the learned weights.

        Raises
        ------
        ValueError
            If the concept probe has not been trained.
        """
        if not self.linear_regressor:
            raise ValueError(
                'The concept probe has not been trained. Call "train_concept_probe" first.'
            )

        weights, _ = self.linear_regressor.layers[0].get_weights()
        vector = weights.flatten()
        return Cav(
            concept_name=self.concept.name, layer_index=self.model_layer_index, vector=vector
        )

    def validate_probe(
        self, validation_dataset: List[np.ndarray], validation_labels: List[np.ndarray]
    ) -> None:
        """
        Evaluates the trained concept probe on an external validation dataset.

        The validation is performed by computing both the R² score and the Mean Squared Error (MSE).

        Parameters
        ----------
        validation_dataset : List[np.ndarray]
            The input activation dataset for validation.
        validation_labels : List[np.ndarray]
            The true concept values for validation.

        Raises
        ------
        ValueError
            If the concept probe has not been trained.
        """
        if not self.linear_regressor:
            raise ValueError(
                'The concept probe has not been trained. Call "train_concept_probe" first.'
            )

        layer_activations = self.model_activation_obtainer.get_layer_activations(
            layer_index=self.model_layer_index,
            model_inputs=np.array(validation_dataset),
            flatten=True,
        )

        y_pred = self.linear_regressor.predict(layer_activations).flatten()

        self.concept_probe_score_on_validation_set = cast(
            float, r2_score(validation_labels, y_pred)
        )
        self.concept_probe_mse_on_validation_set = cast(
            float, mean_squared_error(validation_labels, y_pred)
        )
