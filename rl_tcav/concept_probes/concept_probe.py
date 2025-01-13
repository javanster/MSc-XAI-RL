from abc import ABC, abstractmethod

from ..cav import Cav


class ConceptProbe(ABC):
    """
    Abstract base class for implementing a concept probe.

    A concept probe is a tool for probing a specific layer of a deep neural network (DNN)
    to understand its internal representation of a given concept. Concept probes can take
    various forms, including regressors, classifiers, or other models, and
    are trained to differentiate between concept and non-concept examples. After training,
    the Concept Activation Vector (CAV) can be extracted, which represents the learned
    direction of the concept in the model's feature space.

    Methods
    -------
    train_concept_probe()
        Train the concept probe using labeled examples of the concept and non-concept.
    extract_cav() -> Cav
        Extract the Concept Activation Vector (CAV) after the probe has been trained.
    """

    @abstractmethod
    def train_concept_probe(self) -> None:
        """
        Train the concept probe.

        This method trains the concept probe using labeled examples of the concept
        and non-concept. The training process fits the probe to the provided data,
        learning to differentiate between the concept and non-concept examples.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        pass

    @abstractmethod
    def extract_cav(self) -> Cav:
        """
        Extract the Concept Activation Vector (CAV).

        After training the probe, this method retrieves the CAV that represents
        the direction of the concept in the feature space of a specific layer.

        Returns
        -------
        Cav
            An instance of the `Cav` class containing the concept name, layer index,
            and the CAV vector.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        pass
