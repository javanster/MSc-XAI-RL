import json
from typing import List

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from tqdm import tqdm

from utils import ModelActivationObtainer, ensure_directory_exists


class ICE:
    """
    Invertible Concept-based Explanation (ICE) framework using Non-negative Matrix Factorization (NMF).

    This class enables post hoc concept discovery for CNN models by decomposing spatial feature map
    activations into interpretable non-negative concept activation vectors (NCAVs). It identifies
    concept directions and extracts the top-k highest scoring activation patches for each concept.

    Parameters
    ----------
    last_cnn_layer_idx : int
        Index of the last convolutional layer from which to extract spatial feature maps.
        This should be the last conv layer before any global pooling or dense layers.
    top_k_examples : int
        Number of top-scoring spatial locations to retain per concept for visualization or analysis.
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
        Discover interpretable concepts from CNN feature maps using NMF.

        For each specified number of concepts (`n_concepts`), this method:
        - Extracts feature maps from the given CNN layer
        - Applies NMF to obtain a basis of NCAVs and corresponding concept scores
        - Selects the top-k most activated spatial locations per concept
        - Saves the NCAVs and top-k concept metadata to disk
        - Logs reconstruction error to evaluate the approximation fidelity

        Parameters
        ----------
        observations : np.ndarray
            A batch of input images or observations of shape (n, H, W, C) used to extract
            activations and learn concepts from.
        n_concepts_values : List[int]
            A list of integers, each representing the number of concepts (NMF components) to extract in a run.
            For example, [2, 5, 10, 20] would run NMF four times with different complexity.
        save_dir_path : str
            Directory path to save all outputs. This includes:
            - `*_ncavs.npy`: learned concept direction vectors (NCAVs)
            - `*_topk.json`: metadata for top-k activating positions per concept
            - `stats.csv`: table of reconstruction errors for each `n_concepts` value
        """
        feature_maps = self.mao.get_layer_activations(
            model_inputs=observations, layer_index=self.last_cnn_layer_idx, flatten=False
        )

        n, h, w, c = feature_maps.shape
        V = feature_maps.reshape(-1, c)

        # Track spatial indices so we can trace back to (img, y, x)
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
                    random_state=28,
                    verbose=1,
                    max_iter=500,
                )
                S = nmf_model.fit_transform(V)
                P = nmf_model.components_

                # Save NCAVs
                np.save(f"{save_dir_path}c_{n_concepts}_ncavs.npy", P)

                # Save top-k activations per concept
                concept_topk_info = []
                for concept_id in range(n_concepts):
                    concept_scores = S[:, concept_id]
                    top_indices = np.argsort(concept_scores)[-self.top_k_examples :][::-1]

                    top_k_entries = [
                        {
                            "image_index": int(spatial_indices[idx][0]),
                            "y": int(spatial_indices[idx][1]),
                            "x": int(spatial_indices[idx][2]),
                            "score": float(concept_scores[idx]),
                        }
                        for idx in top_indices
                    ]
                    concept_topk_info.append(
                        {"concept_id": concept_id, "top_k_examples": top_k_entries}
                    )

                with open(f"{save_dir_path}c_{n_concepts}_topk.json", "w") as f:
                    json.dump(concept_topk_info, f, indent=2)

                # Log reconstruction error
                reconstruction = np.linalg.norm(V - S @ P, ord="fro") / np.linalg.norm(V, ord="fro")
                stats.append(
                    {
                        "n_concepts": n_concepts,
                        "reconstruction_error": reconstruction,
                    }
                )

                pbar.update(1)

        df = pd.DataFrame(stats)
        df.to_csv(f"{save_dir_path}stats.csv", index=False)
        print("Saved summary to CSV.")
