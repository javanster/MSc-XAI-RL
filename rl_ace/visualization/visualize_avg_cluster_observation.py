from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def visualize_avg_cluster_observation(
    clustered_observations: np.ndarray,
    clustered_observations_labels: np.ndarray,
    distances: np.ndarray,
    clusters_to_show: List[int],
    show: bool = True,
    save_file_path: str = "",
) -> None:
    """
    Generate and visualize a weighted average image for specified clusters.

    The function computes a weighted average of images in each specified cluster.
    The weights are computed as the inverse of the distances to the cluster centroid,
    emphasizing images that are closer to the centroid. The resulting weighted average
    image for each cluster is then visualized using a grid layout. Plots can be displayed
    or saved based on the provided parameters.

    Parameters
    ----------
    clustered_observations : np.ndarray
        Array of images (observations) with shape either (n_samples, H, W) for grayscale
        images or (n_samples, H, W, C) for color images.
    clustered_observations_labels : np.ndarray
        Array of cluster labels corresponding to each image in `clustered_observations`.
    distances : np.ndarray
        Array of distances of each image to its corresponding cluster centroid.
    clusters_to_show : List[int]
        List of cluster labels for which to generate the weighted average images.
    show : bool, optional
        If True, displays the generated plots. Defaults to True.
    save_file_path : str, optional
        File path prefix used to save the plots. If an empty string is provided, plots will
        not be saved. Defaults to an empty string.

    Raises
    ------
    ValueError
        If the shape of the image array in `clustered_observations` is not 3D or 4D.
    """
    weighted_images = []
    for label in clusters_to_show:
        # Get indices and corresponding images and distances for the current cluster
        cluster_idx = np.where(clustered_observations_labels == label)[0]
        cluster_images = clustered_observations[cluster_idx]
        cluster_distances = distances[cluster_idx]

        # Compute weights: lower distance => higher weight (using inverse distance)
        epsilon = 1e-8  # A small constant to avoid division by zero
        weights = 1.0 / (cluster_distances + epsilon)
        weights /= weights.sum()

        # Reshape weights for broadcasting: handle both grayscale (3D) and color (4D) images.
        if cluster_images.ndim == 4:
            weights = weights[:, None, None, None]
        elif cluster_images.ndim == 3:
            weights = weights[:, None, None]
        else:
            raise ValueError("Unexpected image array shape")

        # Compute the weighted average image for the cluster
        weighted_image = np.sum(cluster_images * weights, axis=0)
        weighted_images.append((weighted_image, label))

    max_images_per_plot = 6
    n_cols = 3
    n_rows = 2

    for start in range(0, len(weighted_images), max_images_per_plot):
        end = start + max_images_per_plot
        batch = weighted_images[start:end]

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for i, (img, label) in enumerate(batch):
            ax = axes[i]
            # Clip the pixel values to valid range and convert to uint8 if needed.
            ax.imshow(np.clip(img, 0, 255).astype(np.uint8))
            ax.set_title(f"Cluster {label}, Weighted Avg")
            ax.axis("off")

        # Hide unused subplots
        for j in range(len(batch), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        if show:
            plt.show()

        if len(save_file_path) > 0:
            batch_labels = [str(label) for _, label in batch]
            batch_labels_str = "_".join(batch_labels)
            plt.savefig(f"{save_file_path}_clusters_{batch_labels_str}.png")

        plt.close()
