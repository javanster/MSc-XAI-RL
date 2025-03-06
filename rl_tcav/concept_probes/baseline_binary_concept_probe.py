from typing import List, cast

import numpy as np
from keras.api.layers import Dense, Input
from keras.api.models import Sequential
from keras.api.optimizers import SGD
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ..cav import Cav
from ..concept_classes.binary_concept import BinaryConcept
from ..model_activation_obtainer import ModelActivationObtainer
from .binary_concept_probe_score import binary_concept_probe_score
from .concept_probe import ConceptProbe


class BaselineBinaryConceptProbe(ConceptProbe):
    """
    A simple binary classification probe for evaluating concept representations.

    This probe learns a linear decision boundary in activation space to classify
    positive and negative examples of a binary concept.

    Parameters
    ----------
    concept : BinaryConcept
        The binary concept to probe.
    model_activation_obtainer : ModelActivationObtainer
        The activation obtainer used to extract activations from the model.
    model_layer_index : int
        The index of the layer from which to obtain activations.

    Attributes
    ----------
    concept : BinaryConcept
        The binary concept associated with this probe.
    binary_classifier : Sequential or None
        The binary classifier model used for probing. None if the probe has not been trained.
    model_activation_obtainer : ModelActivationObtainer
        The activation obtainer used to fetch layer activations.
    model_layer_index : int
        The index of the model layer used for activations.
    concept_probe_score : float or None
        The probe score based on classification performance. None if the probe has not been trained.
    accuracy : float or None
        The accuracy of the probe on the training-validation split. None if the probe has not been trained.
    concept_probe_score_on_validation_set : float or None
        The probe's score when evaluated on an external validation dataset. None if not validated.
    accuracy_on_validation_set : float or None
        The accuracy of the probe when evaluated on an external validation dataset. None if not validated.
    """

    def __init__(
        self,
        concept: BinaryConcept,
        model_activation_obtainer: ModelActivationObtainer,
        model_layer_index: int,
    ) -> None:
        self.concept: BinaryConcept = concept
        self.model_activation_obtainer = model_activation_obtainer
        self.model_layer_index: int = model_layer_index
        self.concept_probe_score: float | None = None
        self.accuracy: float | None = None
        self.concept_probe_score_on_validation_set: float | None = None
        self.accuracy_on_validation_set: float | None = None

    def train_concept_probe(self) -> None:
        """
        Trains the concept probe using a simple binary classifier.

        The activations from the specified model layer are extracted and used to fit
        a logistic regression model predicting binary concept membership.

        Raises
        ------
        ValueError
            If there are fewer than 10 positive or negative concept examples.
        """
        if len(self.concept.positive_examples) < 10 or len(self.concept.negative_examples) < 10:
            raise ValueError(
                "There needs to be a minimum of 10 positive and 10 negative concept examples in order to train a concept probe"
            )

        if len(self.concept.positive_examples) != len(self.concept.negative_examples):
            print(
                "Warning: The number of positive examples is not equal to the number of negative examples. An imbalanced dataset might affect probe training."
            )

        positive_layer_activations = self.model_activation_obtainer.get_layer_activations(
            layer_index=self.model_layer_index,
            model_inputs=np.array(self.concept.positive_examples),
            flatten=True,
        )

        negative_layer_activations = self.model_activation_obtainer.get_layer_activations(
            layer_index=self.model_layer_index,
            model_inputs=np.array(self.concept.negative_examples),
            flatten=True,
        )

        positive_labels = np.ones(positive_layer_activations.shape[0])
        negative_labels = np.zeros(negative_layer_activations.shape[0])

        X = np.vstack([positive_layer_activations, negative_layer_activations])
        y = np.concatenate([positive_labels, negative_labels])

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        input_dim = X_train.shape[1]

        self.binary_classifier = Sequential()
        self.binary_classifier.add(Input(shape=(input_dim,)))
        self.binary_classifier.add(Dense(1, activation="sigmoid"))

        self.binary_classifier.compile(
            optimizer=SGD(learning_rate=0.01),  # type: ignore
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        self.binary_classifier.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)  # type: ignore

        y_pred = self.binary_classifier.predict(X_val)
        y_pred_binary = (y_pred >= 0.5).astype(int)
        self.accuracy = cast(float, accuracy_score(y_val, y_pred_binary))
        self.concept_probe_score = binary_concept_probe_score(y_val=y_val, y_pred=y_pred_binary)

    def extract_cav(self) -> Cav:
        """
        Extracts the Concept Activation Vector (CAV) from the trained probe.

        The CAV is derived from the learned weights of the logistic regression model.

        Returns
        -------
        Cav
            The Concept Activation Vector containing the learned weights.

        Raises
        ------
        ValueError
            If the concept probe has not been trained.
        """
        if not self.binary_classifier:
            raise ValueError(
                'The concept probe has not been trained. Call "train_concept_probe" first.'
            )

        weights, _ = self.binary_classifier.layers[0].get_weights()
        vector = weights.flatten()
        return Cav(
            concept_name=self.concept.name, layer_index=self.model_layer_index, vector=vector
        )

    def validate_probe(
        self, validation_dataset: List[np.ndarray], validation_labels: List[np.ndarray]
    ) -> None:
        """
        Evaluates the trained concept probe on an external validation dataset.

        The validation is performed by computing both accuracy and the concept probe score.

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
        if not self.binary_classifier:
            raise ValueError(
                'The concept probe has not been trained. Call "train_concept_probe" first.'
            )

        layer_activations = self.model_activation_obtainer.get_layer_activations(
            layer_index=self.model_layer_index,
            model_inputs=np.array(validation_dataset),
            flatten=True,
        )

        y_pred = self.binary_classifier.predict(layer_activations)
        y_pred_binary = (y_pred >= 0.5).astype(int)
        self.accuracy_on_validation_set = cast(
            float, accuracy_score(validation_labels, y_pred_binary)
        )
        self.concept_probe_score_on_validation_set = binary_concept_probe_score(
            y_val=np.array(validation_labels), y_pred=y_pred_binary
        )
