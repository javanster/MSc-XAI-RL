from typing import List

import numpy as np
from keras.api.models import Sequential
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from utils import ModelActivationObtainer


class CCM:
    """
    Concept Completeness Model (CCM) to evaluate how well a set of concepts can
    explain the behavior of a neural network.

    This implementation supports both classification (via predicted actions)
    and multi-output regression (via Q-values).

    Parameters
    ----------
    model : Sequential
        The Keras model being explained.
    model_activation_obtainer : ModelActivationObtainer
        Object for extracting model activations.
    num_classes : int
        Number of action classes (used for classification mode).
    X_train : np.ndarray
        Training input observations.
    X_val : np.ndarray
        Validation input observations.
    Y_train : np.ndarray
        Training targets (either action indices or Q-values).
    Y_val : np.ndarray
        Validation targets.
    all_q : bool, optional
        If True, evaluates multi-output regression using Q-values.
        If False, evaluates classification based on discrete actions.
    """

    def __init__(
        self,
        model: Sequential,
        model_activation_obtainer: ModelActivationObtainer,
        num_classes: int,
        X_train: np.ndarray,
        X_val: np.ndarray,
        Y_train: np.ndarray,
        Y_val: np.ndarray,
        all_q: bool = False,
    ) -> None:

        if all_q and np.ndim(Y_train) == 1:
            raise ValueError(
                "Dims of Y_train suggests max Q instead of one Q-value for each action, make sure you have set the correct parameters."
            )

        self.model = model
        self.mao = model_activation_obtainer
        self.num_classes = num_classes
        self.X_train = X_train
        self.X_val = X_val
        self.Y_train = Y_train
        self.Y_val = Y_val
        self.all_q = all_q

    def _adjusted_r2_score(self, Y_val: np.ndarray, Y_pred: np.ndarray, n_predictors: int):
        """
        Compute the adjusted R² score.

        Parameters
        ----------
        Y_val : np.ndarray
            Ground truth target values.
        Y_pred : np.ndarray
            Predicted target values from the model.
        n_predictors : int
            Number of predictor variables (e.g., concept dimensions).

        Returns
        -------
        float
            Adjusted R² score.
        """
        n_samples = Y_val.shape[0]
        r2 = r2_score(Y_val, Y_pred)
        if n_samples <= n_predictors + 1:
            raise ValueError("Number of samples must be grater than CAVs")
        adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_predictors - 1)
        return adjusted_r2

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
        Project model activations onto concept vectors.

        Parameters
        ----------
        model_inputs : np.ndarray
            Input samples to the model.
        cavs : np.ndarray
            Concept Activation Vectors (CAVs), shape: (n_concepts, activation_dim).
        conv_handling : str
            Method for handling convolutional activations ("flatten" or "dim_reduction").
        layer_i : int
            Index of the model layer to extract activations from.

        Returns
        -------
        List[np.ndarray]
            Concept scores for each input sample.
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

    def _train_ccm_model(
        self,
        concept_scores: List[np.ndarray],
    ) -> DecisionTreeClassifier | DecisionTreeRegressor:
        """
        Train a decision tree on concept scores to predict outputs.

        Parameters
        ----------
        concept_scores : List[np.ndarray]
            Concept score representations of the training data.

        Returns
        -------
        DecisionTreeClassifier or DecisionTreeRegressor
            Fitted decision tree model.
        """
        X = np.vstack(concept_scores)
        if self.all_q:
            clf = DecisionTreeRegressor(max_depth=5, random_state=28)
        else:
            clf = DecisionTreeClassifier(max_depth=5, random_state=28)
        clf.fit(X, self.Y_train)

        return clf

    def _evaluate_ccm_model(
        self,
        model: DecisionTreeClassifier | DecisionTreeRegressor,
        cavs: np.ndarray,
        conv_handling: str,
        layer_i: int,
        num_classes: int,
    ) -> float:
        """
        Evaluate the completeness of a concept-based model on validation data.

        Parameters
        ----------
        model : DecisionTreeClassifier or DecisionTreeRegressor
            Trained concept model.
        cavs : np.ndarray
            Concept Activation Vectors (CAVs).
        conv_handling : str
            Method for handling convolutional activations.
        layer_i : int
            Model layer index to extract activations from.
        num_classes : int
            Number of classes (used in classification mode).

        Returns
        -------
        float
            Completeness score: normalized accuracy (classification) or adjusted R² (regression).
        """
        concept_scores = self._get_concept_scores(
            model_inputs=self.X_val, cavs=cavs, conv_handling=conv_handling, layer_i=layer_i
        )

        preds = model.predict(concept_scores)

        if self.all_q:
            n_predictors = cavs.shape[0]  # number of concepts
            completeness_score = float(
                self._adjusted_r2_score(Y_val=self.Y_val, Y_pred=preds, n_predictors=n_predictors)
            )

        else:
            acc = accuracy_score(self.Y_val, preds)
            random_acc = 1 / num_classes
            completeness_score = float((acc - random_acc) / (1 - random_acc))

        return completeness_score

    def train_and_eval_ccm(
        self,
        cavs: np.ndarray,
        conv_handling: str,
        layer_i: int,
    ) -> float:
        """
        Train and evaluate a concept-based decision tree model.

        Parameters
        ----------
        cavs : np.ndarray
            Concept Activation Vectors (CAVs).
        conv_handling : str
            Either "flatten" or "dim_reduction" to control activation handling.
        layer_i : int
            Layer index for extracting activations.

        Returns
        -------
        float
            Completeness score (classification: normalized accuracy, regression: adjusted R²).
        """
        concept_scores = self._get_concept_scores(
            model_inputs=self.X_train, cavs=cavs, conv_handling=conv_handling, layer_i=layer_i
        )
        classifier = self._train_ccm_model(concept_scores=concept_scores)
        completeness_score = self._evaluate_ccm_model(
            model=classifier,
            cavs=cavs,
            conv_handling=conv_handling,
            layer_i=layer_i,
            num_classes=self.num_classes,
        )

        return completeness_score
