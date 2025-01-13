import numpy as np


class Cav:
    """
    A class to represent the Concept Activation Vector (CAV).

    The CAV captures the direction of a concept in the feature space of a specific
    layer of a Deep Neural Network (DNN). It is computed by training a concept probe
    on concept and non-concept examples and is used for interpretability purposes.

    Parameters
    ----------
    concept_name : str
        The name of the concept associated with this CAV.
    layer_index : int
        The index of the DNN layer where the CAV was computed.
    vector : np.ndarray
        The Concept Activation Vector (CAV), typically a 1D array.

    Attributes
    ----------
    concept_name : str
        The name of the concept.
    layer_index : int
        The index of the DNN layer where the CAV was computed.
    vector : np.ndarray
        The Concept Activation Vector (CAV).
    """

    def __init__(
        self,
        concept_name: str,
        layer_index: int,
        vector: np.ndarray,
    ) -> None:
        self.concept_name: str = concept_name
        self.layer_index: int = layer_index
        self.vector: np.ndarray = vector
