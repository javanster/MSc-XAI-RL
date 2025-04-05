from typing import List

import matplotlib.pyplot as plt
import numpy as np


def inspect_cluster_examples(
    clustered_observations: np.ndarray,
    clustered_observations_labels: np.ndarray,
    clusters_to_show: List[int],
    images_to_show: int,
):
    """
    Display example images from specified clusters.

    For each cluster specified in `clusters_to_show`, this function filters the
    `clustered_observations` based on the provided cluster labels and displays
    a grid of example images. The number of images shown per cluster is limited
    by `images_to_show` and the available images in the cluster. Images are displayed
    in batches of 5 per column across the clusters.

    Parameters
    ----------
    clustered_observations : np.ndarray
        Array of images (observations) to be displayed. Each element is expected to be an image.
    clustered_observations_labels : np.ndarray
        Array of cluster labels corresponding to each image in `clustered_observations`.
    clusters_to_show : List[int]
        List of cluster labels for which example images will be displayed.
    images_to_show : int
        The maximum number of images to display from each cluster.

    Returns
    -------
    None
    """
    to_show = []
    for label in clusters_to_show:
        filtered = clustered_observations[clustered_observations_labels == label]
        to_show.append(filtered)

    total_images = min(min(len(obs) for obs in to_show), images_to_show)
    n_clusters = len(to_show)

    for start in range(0, total_images, 5):
        batch_size = min(5, total_images - start)
        _, axes = plt.subplots(batch_size, n_clusters, figsize=(10, 8))
        if batch_size == 1:
            axes = axes[None, :]
        if n_clusters == 1:
            axes = axes[:, None]

        for col, obs in enumerate(to_show):
            for row in range(batch_size):
                ax = axes[row, col]
                ax.imshow(obs[start + row])
                if row == 0:
                    ax.set_title(f"Cluster {clusters_to_show[col] + 1}")
                ax.axis("off")

        plt.tight_layout()
        plt.show()
