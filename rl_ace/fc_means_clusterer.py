from typing import List, Tuple

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from tqdm import tqdm

from utils import ModelActivationObtainer, ensure_directory_exists


class FCMeansClusterer:

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
        if activations.ndim > 2:
            return np.mean(activations, axis=(1, 2))
        else:
            return activations

    def _cluster_activations(
        self,
        activations: np.ndarray,
        c: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Cluster activations using fuzzy c-means clustering.

        Parameters
        ----------
        activations : np.ndarray
            The activation data to be clustered.
        c : int
            The number of clusters to form.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, float]
            A tuple containing:
            - hard_labels: np.ndarray, cluster label with highest membership per point.
            - fuzzy_memberships: np.ndarray, membership values for each point.
            - centroids: np.ndarray, coordinates of cluster centroids.
            - fpc: float, fuzzy partition coefficient (a quality measure).
        """
        data = activations.T
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data,
            c=c,
            m=2,
            error=0.005,
            maxiter=1000,
            init=None,
            seed=28,
        )
        fuzzy_memberships = u.T
        hard_labels = np.argmax(u, axis=0)
        return hard_labels, fuzzy_memberships, cntr, fpc

    def cluster(
        self,
        environment_observations: np.ndarray,
        max_c: int,
        model_layer_indexes: List[int],
        save_directory_path: str,
    ):
        if max_c > len(environment_observations):
            raise ValueError("Cannot form more clusters than number of environment observations")

        stats_list = []

        for layer_i in model_layer_indexes:
            layer_save_directory_path = f"{save_directory_path}layer_{layer_i}/"
            ensure_directory_exists(layer_save_directory_path)

            activations = self.model_activation_obtainer.get_layer_activations(
                layer_index=layer_i, model_inputs=environment_observations, flatten=False
            )

            activations = self._reduce_conv_activations(activations)

            with tqdm(
                initial=2,
                total=max_c,
                unit="c",
                desc=f"Clustering activations from layer {layer_i}",
            ) as pbar:
                for c in range(2, max_c + 1):
                    hard_labels, fuzzy_memberships, centroids, fpc = self._cluster_activations(
                        activations=activations,
                        c=c,
                    )

                    pbar.update(1)

                    np.save(
                        f"{layer_save_directory_path}c_{c}_hard_cluster_labels.npy",
                        hard_labels,
                    )
                    np.save(
                        f"{layer_save_directory_path}c_{c}_fuzzy_memberships.npy",
                        fuzzy_memberships,
                    )
                    np.save(
                        f"{layer_save_directory_path}c_{c}_cluster_centroids.npy",
                        centroids,
                    )

                    stats_list.append(
                        {
                            "layer_index": layer_i,
                            "c": c,
                            "fpc": fpc,
                        }
                    )

        stats_df = pd.DataFrame(stats_list)
        stats_df.to_csv(f"{save_directory_path}/cluster_stats.csv", index=False)
