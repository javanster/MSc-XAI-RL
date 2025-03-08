from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..cav import Cav


class ConceptProbe(ABC):
    """
    Abstract base class for implementing a concept probe.

    A concept probe is a tool for analyzing how a deep neural network (DNN) represents
    a given concept at a specific layer. It is trained on labeled examples and can take
    various forms, such as regression or classification models. After training, the
    Concept Activation Vector (CAV) may be extracted, representing the learned direction
    of the concept in the model's feature space.

    Methods
    -------
    train_concept_probe()
        Train the concept probe using labeled examples of the concept.
    extract_cav() -> Cav
        Extract the Concept Activation Vector (CAV) after training.
    validate_probe(validation_dataset, validation_labels)
        Evaluate the trained probe on an external validation dataset.
    """

    @abstractmethod
    def train_concept_probe(self) -> None:
        """
        Trains the concept probe.

        This method fits the concept probe to labeled examples, learning to map
        activations to concept labels. The training approach depends on the type of
        concept being probed:

        - **Binary Concepts**: The probe learns to classify activations into two categories,
          distinguishing between the presence and absence of the concept.
        - **Continuous Concepts**: The probe learns a regression model that predicts
          the continuous degree of the concept.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        pass

    @abstractmethod
    def extract_cav(self) -> Cav:
        """
        Extracts the Concept Activation Vector (CAV).

        The CAV represents the direction of the concept in the model's feature
        space and is derived from the trained probe.

        Returns
        -------
        Cav
            An instance of the `Cav` class containing the concept name, layer index,
            and the learned CAV vector.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        pass

    @abstractmethod
    def validate_probe(self, validation_dataset: np.ndarray, validation_labels: np.ndarray) -> None:
        """
        Evaluates the trained concept probe on an external validation dataset.

        This method computes evaluation metrics to assess how well the probe generalizes
        to unseen data.

        Parameters
        ----------
        validation_dataset : np.ndarray
            The input activations for validation.
        validation_labels : np.ndarray
            The ground-truth labels for validation.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        pass
