from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from keras.api.layers import Dense, Input
from keras.api.models import Sequential
from keras.api.optimizers import SGD
from sklearn.model_selection import train_test_split

from rl_tcav import binary_concept_probe_score

from .k_selector import KSelector

np.random.seed(28)


class ConceptProbeScoreKSelector(KSelector):
    """
    K-value selector that uses average concept probe score to evaluate clustering quality.

    This class assesses clustering quality by training simple binary probes to distinguish
    each cluster from all others. The classification performance is used to score the
    quality of the concept represented by each cluster. The overall score is a weighted
    average of these individual concept probe scores, weighted by cluster size.

    Attributes
    ----------
    SELECTOR_TYPE : str
        Identifier for the selector type ("average_concept_probe_score").
    score_values : List[float]
        A list of weighted average concept probe scores for each evaluated k value.
    k_values : List[int]
        A list of k values that have been evaluated.

    Methods
    -------
    assess_score_of_cluster(activations, labels, distances_from_centroids, k):
        Computes and stores the weighted average concept probe score for the given clustering result.
    save_scores_plot(save_file_path):
        Saves a plot of probe scores versus k values and highlights the best k.
    get_best_k() -> int:
        Returns the k value that achieved the highest average probe score.
    get_best_score() -> float:
        Returns the highest average concept probe score recorded.
    """

    SELECTOR_TYPE = "average_concept_probe_score"

    def assess_score_of_cluster(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        distances_from_centroids: np.ndarray,
        k: int,
    ) -> None:
        """
        Compute and store the weighted average concept probe score for a clustering result.

        For each cluster, a binary classifier is trained to distinguish that cluster from
        all others. The classification accuracy is treated as a "concept probe score."
        These scores are averaged across all clusters, weighted by cluster size.

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

        unique_concepts = np.unique(labels)
        concept_probe_scores = []
        cluster_sizes = []

        for concept_label in unique_concepts:
            positive_examples = activations[labels == concept_label]
            negative_examples = activations[labels != concept_label]

            set_size = min(len(positive_examples), len(negative_examples))

            if set_size < 30:
                continue  # Not enough data to train a good probe model

            np.random.shuffle(positive_examples)
            np.random.shuffle(negative_examples)

            positive_examples = positive_examples[:set_size]
            negative_examples = negative_examples[:set_size]

            cluster_size = len(positive_examples)

            positive_labels = np.ones(len(positive_examples))
            negative_labels = np.zeros(len(negative_examples))

            X = np.vstack([positive_examples, negative_examples])
            y = np.concatenate([positive_labels, negative_labels])

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            input_dim = X_train.shape[1]
            binary_classifier = Sequential(
                [Input(shape=(input_dim,)), Dense(1, activation="sigmoid")]
            )

            binary_classifier.compile(
                optimizer=SGD(learning_rate=0.01), loss="binary_crossentropy", metrics=["accuracy"]  # type: ignore
            )

            binary_classifier.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)  # type: ignore

            y_pred = binary_classifier.predict(X_val)
            y_pred_binary = (y_pred >= 0.5).astype(int)
            concept_probe_score = binary_concept_probe_score(y_val=y_val, y_pred=y_pred_binary)

            concept_probe_scores.append(concept_probe_score)
            cluster_sizes.append(cluster_size)

        if concept_probe_scores:
            total_weight = np.sum(cluster_sizes)
            weighted_score = float(
                np.sum(np.array(concept_probe_scores) * np.array(cluster_sizes)) / total_weight
            )
            self.score_values.append(weighted_score)
        # If no cluster amounted to at least 30 positive examples and 30 negative examples, append a failure score (-1)
        else:
            self.score_values.append(-1)

        self.k_values.append(k)

    def save_scores_plot(self, save_file_path: str) -> None:
        """
        Save a line plot of average concept probe scores for different k values.

        The plot includes a line showing the weighted average concept probe scores and
        a vertical dashed line marking the k value with the highest score.

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
            label="Average concept probe score for k found concepts",
        )

        best_k = self.get_best_k()
        plt.axvline(x=best_k, color="red", linestyle="--", label=f"Selected k = {best_k}")

        plt.xlabel("Number of Clusters")
        plt.ylabel("Average Concept Probe Score")
        plt.title("Average Concept Probe Score for k Found Concepts")
        plt.legend()
        plt.savefig(save_file_path)
        plt.close()

    def get_best_k(self) -> int:
        """
        Return the k value corresponding to the highest average concept probe score.

        Returns
        -------
        int
            The number of clusters (k) that yielded the highest average probe score.

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
        Return the highest average concept probe score recorded.

        Returns
        -------
        float
            The maximum average probe score among all evaluated k values.

        Raises
        ------
        ValueError
            If no scores have been assessed.
        """
        if not self.score_values:
            raise ValueError("No scores have been assessed yet. Cannot retrieve the best score.")

        return max(self.score_values)
