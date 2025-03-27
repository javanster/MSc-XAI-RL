from abc import ABC, abstractmethod
from typing import List

import numpy as np


class KSelector(ABC):
    """
    Base class for K-value selectors used in clustering.

    This class defines the interface for evaluating clustering configurations
    (i.e., for different values of k) and choosing the optimal k based on
    user-defined scoring or heuristic methods. Subclasses must implement
    the abstract methods to define their own scoring logic, how to plot the
    scores, and how to determine and retrieve the best k.

    Attributes
    ----------
    SELECTOR_TYPE : str
        A string identifier for the selector type (e.g., "elbow_method", "silhouette_method").
    score_values : List[float]
        A list of scores (e.g., inertia, silhouette) for each k tested.
    k_values : List[int]
        A list of k values that have been tested.

    Methods
    -------
    assess_score_of_cluster(activations, labels, distances_from_centroids, k):
        Abstract method to compute and store the score of a particular clustering
        configuration.
    save_scores_plot(save_file_path):
        Abstract method to save a plot of the score values for different k values.
    get_best_k() -> int:
        Abstract method to retrieve the best k value according to the subclass's
        selection strategy.
    get_best_score() -> float:
        Abstract method to retrieve the best (lowest or highest) score value.
    reset():
        Resets the internal lists of scores and k values, so the selector
        can be reused if needed.
    """

    SELECTOR_TYPE = "k_selector_base_class"

    def __init__(self) -> None:
        self.score_values: List[float] = []
        self.k_values: List[int] = []

    def _ensure_k_not_already_assessed(self, k: int) -> None:
        """
        Ensures that a particular k value has not already been evaluated.

        Parameters
        ----------
        k : int
            The number of clusters to check for duplication.

        Raises
        ------
        ValueError
            If k has already been assessed previously.
        """
        if k in self.k_values:
            raise ValueError("This value of k has already been assessed!")

    @abstractmethod
    def assess_score_of_cluster(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        distances_from_centroids: np.ndarray,
        k: int,
    ) -> None:
        """
        Abstract method to compute and record the score of a clustering configuration.

        Parameters
        ----------
        activations : np.ndarray
            The matrix of activations or feature vectors.
        labels : np.ndarray
            The cluster labels assigned to each sample in activations.
        distances_from_centroids : np.ndarray
            The distances of each sample to the centroid of its assigned cluster.
        k : int
            The number of clusters used in the clustering.
        """
        pass

    @abstractmethod
    def save_scores_plot(self, save_file_path: str) -> None:
        """
        Abstract method to save a plot showing the scores for different k values.

        Parameters
        ----------
        save_file_path : str
            The file path where the plot image will be saved.
        """
        pass

    @abstractmethod
    def get_best_k(self) -> int:
        """
        Abstract method to return the best k value based on the stored scores.

        Returns
        -------
        int
            The optimal k value determined by the subclass's selection method.
        """
        pass

    @abstractmethod
    def get_best_score(self) -> float:
        """
        Abstract method to return the score corresponding to the best k value.

        Returns
        -------
        float
            The best score among the stored score values.
        """
        pass

    def reset(self) -> None:
        """
        Reset the recorded k values and score values so that the selector
        can be used from scratch.
        """
        self.score_values = []
        self.k_values = []
