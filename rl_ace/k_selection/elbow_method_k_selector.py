import matplotlib.pyplot as plt
import numpy as np

from .k_selector import KSelector


class ElbowMethodKSelector(KSelector):
    """
    KSelector subclass that selects the best k using a 'distance from chord' approach.
    Calculates inertia based on distances between activations and centroids.
    """

    SELECTOR_TYPE = "elbow_method"

    def assess_score_of_cluster(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        distances_from_centroids: np.ndarray,
        k: int,
    ) -> None:
        """
        Assess the score of a clustering configuration using calculated inertia.

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

        # Calculate inertia as the sum of squared distances
        inertia = float(np.sum(distances_from_centroids**2))

        self.score_values.append(inertia)
        self.k_values.append(k)

    def save_scores_plot(self, save_file_path: str) -> None:
        """
        Save the Elbow plot showing inertia for different values of k and highlight the best k.

        Parameters
        ----------
        save_file_path : str
            The file path to save the generated plot.
        """
        if not self.k_values:
            raise ValueError("No scores have been assessed yet. Cannot generate plot.")

        plt.figure(figsize=(8, 6))
        plt.plot(
            self.k_values,
            self.score_values,
            marker="o",
            color="blue",
            label="Inertia (Elbow Method)",
        )
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia (Sum of Squared Distances)")
        plt.title("Elbow Method Analysis for Different Values of K")
        plt.legend()

        # Highlight the best k
        best_k = self.get_best_k()
        best_score = self.get_best_score()

        plt.scatter(best_k, best_score, color="red", s=100, label=f"Best k = {best_k}")
        plt.legend()

        # Save the plot
        plt.savefig(save_file_path)
        plt.close()

    def get_best_k(self) -> int:
        """
        Identify the best k using the 'distance from chord' approach.

        Returns
        -------
        int
            The optimal number of clusters.
        """
        if not self.k_values:
            raise ValueError("No scores have been assessed yet. Cannot determine the best k.")
        if len(self.k_values) < 2:
            # If we only have one or no k-values, we cannot form a chord
            return self.k_values[0]

        # Convert lists to numpy arrays
        k_vals = np.array(self.k_values)
        inertia_vals = np.array(self.score_values)

        # The first point is (k_vals[0], inertia_vals[0])
        # The last point is (k_vals[-1], inertia_vals[-1])
        # We will measure the distance of each intermediate point to the line
        x1, y1 = k_vals[0], inertia_vals[0]
        x2, y2 = k_vals[-1], inertia_vals[-1]

        # Vector from first to last point
        line_vec = np.array([x2 - x1, y2 - y1])
        line_len = np.sqrt(line_vec[0] ** 2 + line_vec[1] ** 2)

        # If the line length is 0, it means all inertia values are identical
        if line_len == 0:
            # Just pick the smallest k in that case
            return int(k_vals[0])

        # Distances of each point (k, inertia) to the line
        distances = []
        for i in range(len(k_vals)):
            # Current point
            x0, y0 = k_vals[i], inertia_vals[i]

            # Vector from first point to current point
            pt_vec = np.array([x0 - x1, y0 - y1])

            # Cross product's magnitude gives area of parallelogram
            # area = |line_vec x pt_vec|
            # distance = area / base
            cross = np.abs(line_vec[0] * pt_vec[1] - line_vec[1] * pt_vec[0])
            dist = cross / line_len
            distances.append(dist)

        # Find the index with the maximum distance
        elbow_index = np.argmax(distances)
        return int(k_vals[elbow_index])

    def get_best_score(self) -> float:
        """
        Get the score (inertia) corresponding to the best k (elbow point).

        Returns
        -------
        float
            The score (inertia) at the best k.
        """
        if not self.score_values:
            raise ValueError("No scores have been assessed yet. Cannot retrieve the best score.")

        best_k_index = self.k_values.index(self.get_best_k())
        return self.score_values[best_k_index]
