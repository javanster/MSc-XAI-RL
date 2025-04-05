from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

from rl_tcav import ModelActivationObtainer
from utils import ensure_directory_exists


class KMeansClusterer:
    """
    A class for clustering model activations using KMeans clustering.

    This class obtains activations for specified model layers using a provided
    ModelActivationObtainer instance, reduces the dimensionality of convolutional
    activations if needed, and performs KMeans clustering over a range of cluster
    counts. It saves the cluster labels for each k and computes various statistics
    about the clustering.

    Parameters
    ----------
    model_activation_obtainer : ModelActivationObtainer
        An instance that provides activations from a model for given inputs.
    """

    def __init__(self, model_activation_obtainer: ModelActivationObtainer) -> None:
        self.model_activation_obtainer = model_activation_obtainer

    def _reduce_conv_activations(self, activations: np.ndarray) -> np.ndarray:
        """
        Reduce convolutional activations by averaging over spatial dimensions if applicable.

        Parameters
        ----------
        activations : np.ndarray
            The activations to be reduced. If the array has more than 2 dimensions
            (e.g., shape (batch, H, W, C)), the spatial dimensions (H, W) will be averaged.

        Returns
        -------
        np.ndarray
            The reduced activations with spatial dimensions averaged if necessary.
        """
        # Check if activations have more than 2 dimensions (batch, H, W, C)
        if activations.ndim > 2:
            print(
                f"Activations shape {activations.shape} detected. Averaging over spatial dimensions."
            )
            return np.mean(activations, axis=(1, 2))
        else:
            return activations

    def _cluster_activations(
        self,
        activations: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster activations using KMeans clustering.

        Parameters
        ----------
        activations : np.ndarray
            The activation data to be clustered.
        k : int
            The number of clusters to form.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - labels: np.ndarray, cluster labels for each activation.
            - distances: np.ndarray, the Euclidean distances of each activation to its
              corresponding cluster centroid.
        """
        kmeans = KMeans(n_clusters=k, random_state=28)
        labels = kmeans.fit_predict(activations)

        distances = np.linalg.norm(activations - kmeans.cluster_centers_[labels], axis=1)

        return labels, distances

    def cluster(
        self,
        environment_observations: np.ndarray,
        max_k: int,
        model_layer_indexes: List[int],
        save_directory_path: str,
    ):
        """
        Cluster activations across multiple model layers and a range of cluster counts.

        For each layer index provided in `model_layer_indexes`, this method obtains
        the corresponding layer activations from the model, reduces the activations if
        necessary, and performs KMeans clustering for each number of clusters from 2 to
        `max_k`. The method saves the cluster labels for each configuration and compiles
        clustering statistics into a CSV file.

        Parameters
        ----------
        environment_observations : np.ndarray
            The input observations for which model activations are to be obtained.
        max_k : int
            The maximum number of clusters to evaluate. Must be less than or equal to
            the number of environment observations.
        model_layer_indexes : List[int]
            A list of model layer indexes from which to extract activations.
        save_directory_path : str
            The directory path where clustering results and statistics will be saved.

        Raises
        ------
        ValueError
            If `max_k` is greater than the number of environment observations.
        """
        if max_k > len(environment_observations):
            raise ValueError("Cannot form more clusters than number of environment observations")

        stats_list = []

        for layer_i in model_layer_indexes:
            layer_save_directory_path = f"{save_directory_path}layer_{layer_i}/"

            activations = self.model_activation_obtainer.get_layer_activations(
                layer_index=layer_i, model_inputs=environment_observations, flatten=False
            )

            activations = self._reduce_conv_activations(activations)

            with tqdm(
                initial=2,
                total=max_k,
                unit="k",
                desc=f"Clustering activations from layer {layer_i}",
            ) as pbar:
                for k in range(2, max_k + 1):

                    k_save_directory_path = f"{layer_save_directory_path}/k_{k}/"
                    ensure_directory_exists(directory_path=k_save_directory_path)

                    cluster_labels, centroid_distances = self._cluster_activations(
                        activations=activations,
                        k=k,
                    )

                    pbar.update(1)

                    np.save(
                        f"{k_save_directory_path}/layer_{layer_i}_k_{k}_cluster_labels.npy",
                        cluster_labels,
                    )

                    cluster_counts = Counter(cluster_labels)
                    num_observations_per_cluster = np.array(list(cluster_counts.values()))

                    min_observations = num_observations_per_cluster.min()
                    max_observations = num_observations_per_cluster.max()
                    avg_observations = num_observations_per_cluster.mean()
                    std_observations = num_observations_per_cluster.std()
                    avg_distance_to_centroid = np.mean(centroid_distances)
                    total_observations = len(cluster_labels)
                    median_distance_to_centroid = np.median(centroid_distances)
                    inertia = np.sum(centroid_distances**2)
                    size_variation_ratio = (
                        max_observations / min_observations if min_observations > 0 else np.nan
                    )

                    stats_list.append(
                        {
                            "layer_index": layer_i,
                            "k": k,
                            "total_observations_in_all_clusters": total_observations,
                            "min_observation_n_in_any_cluster": min_observations,
                            "max_observation_n_in_any_cluster": max_observations,
                            "avg_observation_n_in_all_clusters": avg_observations,
                            "std_observation_n_in_all_clusters": std_observations,
                            "avg_distance_to_centroid": avg_distance_to_centroid,
                            "median_distance_to_centroid": median_distance_to_centroid,
                            "inertia": inertia,
                            "cluster_size_variation_ratio": size_variation_ratio,
                        }
                    )

        stats_df = pd.DataFrame(stats_list)
        stats_df.to_csv(f"{save_directory_path}/cluster_stats.csv", index=False)
