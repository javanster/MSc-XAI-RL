import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
from keras.api.models import Sequential
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from utils import ModelActivationObtainer

ConceptCompletenessModel = Union[DecisionTreeClassifier, DecisionTreeRegressor, Sequential]


class CCM(ABC):
    """
    Abstract base class for Concept Completeness Models (CCM).

    This class provides a common interface for training and evaluating models that
    assess the completeness of a set of concept activation vectors (CAVs) in
    explaining the behavior of a neural network.

    Subclasses must implement the `_train_ccm_model` method, which defines how
    the concept-based model (e.g., decision tree, neural network) is trained.

    Parameters
    ----------
    model_activation_obtainer : ModelActivationObtainer
        Object for extracting intermediate activations from the model to be explained.
    num_classes : int
        Number of classes (used for classification mode).
    X_train : np.ndarray
        Training input observations.
    X_val : np.ndarray
        Validation input observations.
    Y_train : np.ndarray
        Training targets (either action indices or Q-values).
    Y_val : np.ndarray
        Validation targets.
    all_q : bool, optional
        If True, evaluates regression (Q-values); otherwise classification.
    """

    def __init__(
        self,
        model_activation_obtainer: ModelActivationObtainer,
        num_classes: int,
        X_train: np.ndarray,
        X_val: np.ndarray,
        Y_train: np.ndarray,
        Y_val: np.ndarray,
        all_q: bool = False,
    ) -> None:
        random.seed(28)
        np.random.seed(28)
        tf.random.set_seed(28)

        if all_q and np.ndim(Y_train) == 1:
            raise ValueError(
                "Dims of Y_train suggests max Q instead of one Q-value for each action, make sure you have set the correct parameters."
            )

        self.mao = model_activation_obtainer
        self.num_classes = num_classes
        self.X_train = X_train
        self.X_val = X_val
        self.Y_train = Y_train
        self.Y_val = Y_val
        self.all_q = all_q

    def _dim_reduce_conv_activations(self, activations: np.ndarray) -> np.ndarray:
        """
        Reduce convolutional activations using global average pooling, as done in ACE.

        Parameters
        ----------
        activations : np.ndarray
            Raw activations from a convolutional layer (shape: [batch, H, W, C]).

        Returns
        -------
        np.ndarray
            Reduced activations (shape: [batch, C]).
        """
        if activations.ndim > 2:
            return np.mean(activations, axis=(1, 2))
        else:
            return activations

    def _get_concept_scores(
        self,
        model_inputs: np.ndarray,
        cavs: np.ndarray,
        conv_handling: str,
        layer_i: int,
    ) -> List[np.ndarray]:
        """
        Project model activations onto concept vectors to compute concept scores.

        Parameters
        ----------
        model_inputs : np.ndarray
            Input samples to the model.
        cavs : np.ndarray
            Concept Activation Vectors (CAVs), shape: (n_concepts, activation_dim).
        conv_handling : str
            How to handle conv activations: "flatten" or "dim_reduction".
        layer_i : int
            Index of the model layer to extract activations from.

        Returns
        -------
        List[np.ndarray]
            Concept score vectors for each input sample.
        """
        if conv_handling == "flatten":
            activations = self.mao.get_layer_activations(
                layer_index=layer_i, model_inputs=model_inputs, flatten=True
            )
        else:
            activations = self.mao.get_layer_activations(
                layer_index=layer_i, model_inputs=model_inputs, flatten=False
            )
            activations = self._dim_reduce_conv_activations(activations=activations)
        cavs_transposed = cavs.T
        assert (
            activations.shape[1] == cavs_transposed.shape[0]
        ), f"Shape mismatch: activations {activations.shape}, cavs {cavs_transposed.shape}"
        concept_scores = np.dot(activations, cavs_transposed)

        return concept_scores

    @abstractmethod
    def _train_ccm_model(
        self,
        concept_scores: List[np.ndarray],
    ) -> ConceptCompletenessModel:
        """
        Abstract method for training a model on concept scores.

        Parameters
        ----------
        concept_scores : List[np.ndarray]
            List of concept score vectors for each training sample.

        Returns
        -------
        ConceptCompletenessModel
            A trained model (e.g., decision tree, neural network) that maps concept scores to outputs.
        """
        pass

    def _evaluate_ccm_model(
        self,
        ccm: ConceptCompletenessModel,
        cavs: np.ndarray,
        conv_handling: str,
        layer_i: int,
    ) -> float:
        """
        Evaluate completeness on the validation set using a trained concept-based model.

        Parameters
        ----------
        ccm : ConceptCompletenessModel
            Trained model (classifier or regressor) for predicting from concept scores.
        cavs : np.ndarray
            Concept Activation Vectors.
        conv_handling : str
            How to handle conv activations: "flatten" or "dim_reduction".
        layer_i : int
            Model layer index from which to extract activations.

        Returns
        -------
        float
            Completeness score:
            - Normalized accuracy for classification,
            - R² score for regression.
        """
        concept_scores = self._get_concept_scores(
            model_inputs=self.X_val, cavs=cavs, conv_handling=conv_handling, layer_i=layer_i
        )

        preds = ccm.predict(concept_scores)

        if self.all_q:
            completeness_score = float(r2_score(self.Y_val, preds))

        else:
            acc = accuracy_score(self.Y_val, preds)
            random_acc = 1 / self.num_classes
            completeness_score = float((acc - random_acc) / (1 - random_acc))

        return completeness_score

    def train_and_eval_ccm(
        self,
        cavs: np.ndarray,
        conv_handling: str,
        layer_i: int,
    ) -> Tuple[float, ConceptCompletenessModel]:
        """
        Train the concept model and evaluate its completeness on validation data.

        Parameters
        ----------
        cavs : np.ndarray
            Concept Activation Vectors.
        conv_handling : str
            Either "flatten" or "dim_reduction" for activation handling.
        layer_i : int
            Layer index from which to extract activations.

        Returns
        -------
        Tuple[float, ConceptCompletenessModel]
            - Completeness score,
            - Trained model.
        """
        concept_scores = self._get_concept_scores(
            model_inputs=self.X_train, cavs=cavs, conv_handling=conv_handling, layer_i=layer_i
        )
        ccm = self._train_ccm_model(concept_scores=concept_scores)
        completeness_score = self._evaluate_ccm_model(
            ccm=ccm,
            cavs=cavs,
            conv_handling=conv_handling,
            layer_i=layer_i,
        )

        return completeness_score, ccm
