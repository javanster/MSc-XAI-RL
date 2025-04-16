import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from sklearn.cluster import KMeans

from .k_selector import KSelector


class GapStatisticKSelector(KSelector):
    """
    KSelector subclass that selects the best k using the Gap Statistic method.
    Compares inertia of actual data with reference data to determine optimal k.
    """

    SELECTOR_TYPE = "gap_statistic"

    def __init__(self, n_references: int = 10) -> None:
        """
        Initialize the GapStatisticKSelector.

        Parameters
        ----------
        n_references : int
            The number of random reference datasets to generate for calculation of the gap statistic.
        """
        super().__init__()
        self.n_references = n_references  # Number of random datasets to compare against

    def assess_score_of_cluster(
        self, activations: ndarray, labels: ndarray, distances_from_centroids: ndarray, k: int
    ) -> None:
        """
        Assess the score of a clustering configuration using the Gap Statistic.

        Parameters
        ----------
        activations : ndarray
            The activations or feature matrix to cluster.
        labels : ndarray
            The cluster labels assigned to the activations.
        distances_from_centroids : ndarray
            The distances of each point to its nearest centroid.
        k : int
            The number of clusters used to compute the inertia.
        """
        self._ensure_k_not_already_assessed(k=k)

        # Calculate observed inertia
        observed_inertia = float(np.sum(distances_from_centroids**2))

        # Generate reference datasets and calculate their inertia
        reference_inertias = []
        for _ in range(self.n_references):
            random_data = np.random.uniform(
                np.min(activations, axis=0), np.max(activations, axis=0), size=activations.shape
            )
            kmeans = KMeans(n_clusters=k, random_state=28)
            kmeans.fit(random_data)
            reference_inertias.append(kmeans.inertia_)

        # Compute the gap statistic
        log_observed_inertia = np.log(observed_inertia)
        log_reference_inertia = np.mean(np.log(reference_inertias))
        gap_value = log_reference_inertia - log_observed_inertia

        # Store the gap value as the score
        self.score_values.append(gap_value)
        self.k_values.append(k)

    def save_scores_plot(self, save_file_path: str) -> None:
        """
        Save the Gap Statistic plot for different values of k.

        Parameters
        ----------
        save_file_path : str
            The file path to save the generated plot.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(
            self.k_values,
            self.score_values,
            marker="o",
            color="green",
            label="Gap Statistic",
        )
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Gap Value")
        plt.title("Gap Statistic Analysis for Different Values of K")
        plt.legend()

        plt.savefig(save_file_path)
        plt.close()

    def get_best_k(self) -> int:
        """
        Identify the best k using the maximum Gap value.

        Returns
        -------
        int
            The optimal number of clusters.
        """
        if not self.k_values:
            raise ValueError("No scores have been assessed yet. Cannot determine the best k.")

        # The best k is the one with the highest Gap value
        return self.k_values[np.nanargmax(self.score_values)]

    def get_best_score(self) -> float:
        """
        Get the best Gap value (score) corresponding to the best k.

        Returns
        -------
        float
            The highest gap value.
        """
        if not self.score_values:
            raise ValueError("No scores have been assessed yet. Cannot retrieve the best score.")

        return max(self.score_values)
