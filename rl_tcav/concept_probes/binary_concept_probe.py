from typing import cast

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ..cav import Cav
from ..concept_classes.binary_concept import BinaryConcept
from ..model_activation_obtainer import ModelActivationObtainer
from .binary_concept_probe_score import binary_concept_probe_score
from .concept_probe import ConceptProbe


class BinaryConceptProbe(ConceptProbe):
    """
    WORK IN PROGRESS
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
        self.binary_classifier: LogisticRegression | None = None
        self.concept_probe_score: float | None = None

    def train_concept_probe(self) -> None:
        if len(self.concept.positive_examples) < 20 or len(self.concept.negative_examples) < 20:
            raise ValueError(
                "There needs to be a minimum of 20 positive and negative concept examples in order to train a concept probe"
            )

        if len(self.concept.positive_examples) != len(self.concept.negative_examples):
            print(
                "Warning: The number of positive examples is not equal to the number of negative examples. An imbalanced dataset might affect probe training."
            )

        positive_layer_activations = self.model_activation_obtainer.get_layer_activations(
            layer_index=self.model_layer_index,
            model_inputs=np.array(self.concept.positive_examples),
        )

        negative_layer_activations = self.model_activation_obtainer.get_layer_activations(
            layer_index=self.model_layer_index,
            model_inputs=np.array(self.concept.negative_examples),
        )

        positive_labels = np.ones(positive_layer_activations.shape[0])
        negative_labels = np.zeros(negative_layer_activations.shape[0])

        X = np.vstack([positive_layer_activations, negative_layer_activations])
        y = np.concatenate([positive_labels, negative_labels])

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.binary_classifier = LogisticRegression(max_iter=2000, n_jobs=-1, solver="saga")
        self.binary_classifier.fit(X=X_train, y=y_train)

        y_pred = self.binary_classifier.predict(X_val)
        self.concept_probe_score = binary_concept_probe_score(y_val=y_val, y_pred=y_pred)

    def extract_cav(self) -> Cav:
        if not self.binary_classifier:
            raise ValueError(
                'The concept probe has not been trained. Call "train_concept_probe" first.'
            )

        vector = self.binary_classifier.coef_[0]
        return Cav(
            concept_name=self.concept.name, layer_index=self.model_layer_index, vector=vector
        )
