import os
import pickle
from typing import List

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from tqdm import tqdm

from utils import ModelActivationObtainer, ensure_directory_exists


class ICE:
    """
    Invertible Concept-based Explanation (ICE) framework using Non-negative Matrix Factorization (NMF).

    This class implements a post hoc concept discovery method for convolutional neural networks.
    It decomposes spatial feature map activations into interpretable non-negative concept activation
    vectors (NCAVs) and extracts the top-k highest scoring activation patches for each concept.

    Parameters
    ----------
    last_cnn_layer_idx : int
        Index of the last convolutional layer to extract feature maps. This should be the last conv layer
        before any global pooling or dense layers.
    top_k_examples : int
        Number of top-scoring images to retain per concept for visualization or analysis.
    model_activation_obtainer : ModelActivationObtainer
        Utility object that handles extraction of intermediate layer activations from a Keras model.
    """

    def __init__(
        self,
        last_cnn_layer_idx: int,
        top_k_examples: int,
        model_activation_obtainer: ModelActivationObtainer,
    ) -> None:
        self.last_cnn_layer_idx = last_cnn_layer_idx
        self.top_k_examples = top_k_examples
        self.mao = model_activation_obtainer

    def discover_concepts(
        self, observations: np.ndarray, n_concepts_values: List[int], save_dir_path: str
    ):
        """
        Discover interpretable concepts from CNN feature maps using Non-negative Matrix Factorization (NMF).

        For each specified number of concepts, this method:
        - Extracts spatial feature maps from a specified convolutional layer.
        - Applies NMF to factorize the activations into NCAVs and concept activations.
        - Selects the top-k distinct images for each concept based on the highest activations.
        - Constructs normalized (0-1) 2D activation maps for selected images and stores them in a dictionary.
        - Saves NCAVs and activation maps to disk for further inspection.
        - Computes and stores reconstruction error statistics for each number of concepts.

        Parameters
        ----------
        observations : np.ndarray
            Input data to be fed into the model for obtaining layer activations. Shape is typically (n_samples, height, width, channels).
        n_concepts_values : List[int]
            List of values for the number of concepts to discover (i.e., NMF components).
        save_dir_path : str
            Path to the directory where NCAVs, activation maps, and statistics will be saved.

        Returns
        -------
        None
        """

        # 1. Extract feature maps
        feature_maps = self.mao.get_layer_activations(
            model_inputs=observations, layer_index=self.last_cnn_layer_idx, flatten=False
        )

        n, h, w, c = feature_maps.shape
        V = feature_maps.reshape(-1, c)

        # Track which (image, y, x) each row of V corresponds to
        spatial_indices = np.array(
            [(i, y, x) for i in range(n) for y in range(h) for x in range(w)]
        )

        stats = []
        ensure_directory_exists(save_dir_path)

        with tqdm(total=len(n_concepts_values), unit="c") as pbar:
            for n_concepts in n_concepts_values:
                nmf_model = NMF(
                    n_components=n_concepts,
                    init="random",
                    random_state=None,
                    verbose=1,
                    max_iter=500,
                )
                S = nmf_model.fit_transform(V)  # shape: (n*h*w, n_concepts)
                P = nmf_model.components_  # shape: (n_concepts, c)

                ncavs_filename = f"c_{n_concepts}_ncavs.npy"
                np.save(os.path.join(save_dir_path, ncavs_filename), P)

                concept_activations_dict = {}

                for concept_id in range(n_concepts):
                    # Extract scores for this concept across all spatial positions.
                    concept_scores = S[:, concept_id]

                    # Compute average activation per image in a vectorized manner.
                    # spatial_indices[:, 0] holds the image indices for each row in V.
                    # The number of spatial positions per image is h*w.
                    img_indices_all = spatial_indices[:, 0]
                    avg_activation_per_image = np.bincount(
                        img_indices_all, weights=concept_scores
                    ) / (h * w)

                    # Obtain the indices sorted in descending order by concept activation.
                    sorted_indices = np.argsort(concept_scores)[::-1]

                    unique_top_indices = []
                    seen_images = set()
                    # Iterate over all sorted indices until we have self.top_k_examples unique image entries.
                    for idx in sorted_indices:
                        img_idx = int(spatial_indices[idx][0])
                        if img_idx not in seen_images:
                            unique_top_indices.append(idx)
                            seen_images.add(img_idx)
                        if len(unique_top_indices) >= self.top_k_examples:
                            break

                    topk_list_for_this_concept = []
                    for idx in unique_top_indices:
                        img_idx = int(spatial_indices[idx][0])
                        # Get all spatial rows corresponding to this image
                        img_spatial_indices = np.where(spatial_indices[:, 0] == img_idx)[0]
                        # Reshape the concept scores for the entire image to (h, w)
                        activation_map = concept_scores[img_spatial_indices].reshape(h, w)

                        # min–max normalize the activation map so values are between 0 and 1.
                        norm_activation_map = activation_map - activation_map.min()
                        if norm_activation_map.max() > 0:
                            norm_activation_map /= norm_activation_map.max()

                        record = {
                            "image_index": img_idx,
                            "activation_map": norm_activation_map,
                            "avg_activation": float(avg_activation_per_image[img_idx]),
                        }
                        topk_list_for_this_concept.append(record)

                    concept_activations_dict[concept_id] = topk_list_for_this_concept

                topk_acts_filename = f"c_{n_concepts}_topk_activations.pkl"
                with open(os.path.join(save_dir_path, topk_acts_filename), "wb") as f:
                    pickle.dump(concept_activations_dict, f)

                reconstruction_error = np.linalg.norm(V - S @ P, ord="fro") / np.linalg.norm(
                    V, ord="fro"
                )
                stats.append(
                    {
                        "n_concepts": n_concepts,
                        "reconstruction_error": reconstruction_error,
                    }
                )

                pbar.update(1)

        df = pd.DataFrame(stats)
        df.to_csv(os.path.join(save_dir_path, "stats.csv"), index=False)
