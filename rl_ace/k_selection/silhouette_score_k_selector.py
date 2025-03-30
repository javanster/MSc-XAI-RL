import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score

from .k_selector import KSelector


class SilhouetteScoreKSelector(KSelector):
    """
    K-value selector that uses the Silhouette Score to evaluate clustering quality.

    This class implements the silhouette score as a metric to assess the quality of
    clustering configurations for different values of k. It inherits from `KSelector`
    and provides concrete implementations for computing scores, determining the best k,
    and visualizing the results.

    Attributes
    ----------
    SELECTOR_TYPE : str
        Identifier for the selector type ("silhouette_score").
    score_values : List[float]
        A list of silhouette scores computed for each evaluated k value.
    k_values : List[int]
        A list of k values that have been evaluated.

    Methods
    -------
    assess_score_of_cluster(activations, labels, distances_from_centroids, k):
        Computes and stores the silhouette score for the given clustering result.
    save_scores_plot(save_file_path):
        Saves a plot of silhouette scores versus k values and highlights the best k.
    get_best_k() -> int:
        Returns the k value that achieved the highest silhouette score.
    get_best_score() -> float:
        Returns the highest silhouette score recorded.
    """

    SELECTOR_TYPE = "silhouette_score"

    def assess_score_of_cluster(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        distances_from_centroids: np.ndarray,
        k: int,
    ) -> None:
        """
        Compute and store the silhouette score for a given clustering result.

        Parameters
        ----------
        activations : np.ndarray
            The matrix of activations or feature vectors.
        labels : np.ndarray
            Cluster labels assigned to each sample in the activations.
        distances_from_centroids : np.ndarray
            Distances of each sample to the centroid of its assigned cluster
            (not used in this implementation but required by the interface).
        k : int
            The number of clusters used in the clustering.
        """
        self._ensure_k_not_already_assessed(k=k)
        score = float(silhouette_score(activations, labels))
        self.score_values.append(score)
        self.k_values.append(k)

    def save_scores_plot(self, save_file_path: str) -> None:
        """
        Save a line plot of silhouette scores for different k values.

        The plot includes a line showing the silhouette scores and a vertical dashed
        line marking the k value with the highest score.

        Parameters
        ----------
        save_file_path : str
            The file path where the plot image will be saved.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(
            self.k_values,
            self.score_values,
            marker="o",
            color="orange",
            label="Silhouette Score",
        )

        best_k = self.get_best_k()

        plt.axvline(x=best_k, color="red", linestyle="--", label=f"Selected k = {best_k}")

        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Analysis for Different Values of K")
        plt.legend()

        plt.savefig(save_file_path)
        plt.close()

    def get_best_k(self) -> int:
        """
        Return the k value corresponding to the highest silhouette score.

        Returns
        -------
        int
            The number of clusters (k) that yielded the highest silhouette score.

        Raises
        ------
        ValueError
            If no scores have been assessed.
        """
        if not self.k_values:
            raise ValueError("No scores have been assessed yet. Cannot determine the best k.")

        return self.k_values[np.nanargmax(self.score_values)]

    def get_best_score(self) -> float:
        """
        Return the highest silhouette score recorded.

        Returns
        -------
        float
            The maximum silhouette score among all evaluated k values.

        Raises
        ------
        ValueError
            If no scores have been assessed.
        """
        if not self.score_values:
            raise ValueError("No scores have been assessed yet. Cannot retrieve the best score.")

        return max(self.score_values)
