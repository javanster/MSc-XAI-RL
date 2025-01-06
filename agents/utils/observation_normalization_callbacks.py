from typing import Callable, Dict

import numpy as np


class ObservationNormalizationCallbacks:
    """
    A collection of static methods and pre-defined callbacks for normalizing observations.

    This class provides utility functions for normalizing observations in reinforcement learning environments.
    It also maintains a dictionary of normalization callbacks for easy selection and extensibility.

    Attributes
    ----------
    normalization_callbacks : Dict[str, Callable[[np.ndarray], np.ndarray]]
        A dictionary mapping normalization types (keys) to their respective callback functions (values).

    Methods
    -------
    non_normalize(observations: np.ndarray) -> np.ndarray
        Returns the observations without applying any normalization.

    normalize_images(observations: np.ndarray) -> np.ndarray
        Normalizes image-like observations by scaling pixel values to the range [0, 1].
    """

    @staticmethod
    def non_normalize(observations: np.ndarray) -> np.ndarray:
        """
        Returns the observations without applying any normalization.

        Parameters
        ----------
        observations : np.ndarray
            The raw observations from the environment.

        Returns
        -------
        np.ndarray
            The unmodified observations.
        """
        return observations

    @staticmethod
    def normalize_images(observations: np.ndarray) -> np.ndarray:
        """
        Normalizes image-like observations by scaling pixel values to the range [0, 1].

        Parameters
        ----------
        observations : np.ndarray
            The raw image-like observations, typically with pixel values in the range [0, 255].

        Returns
        -------
        np.ndarray
            The normalized observations with pixel values in the range [0, 1].
        """
        return observations / 255

    normalization_callbacks: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "no_normalization": non_normalize,
        "image": normalize_images,
    }
