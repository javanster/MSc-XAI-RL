from typing import List

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from utils import ModelActivationObtainer

from .ccm import CCM


class CCM_DT(CCM):
    """
    Concept Completeness Model using a Decision Tree decoder.

    This subclass of CCM uses a decision tree (classifier or regressor)
    to model the relationship between concept scores and model predictions.
    It supports both classification (using action indices) and regression
    (using Q-values) depending on the `all_q` flag.

    Parameters
    ----------
    model_activation_obtainer : ModelActivationObtainer
        Utility to extract intermediate layer activations from the model to be explained.
    num_classes : int
        Number of output classes for classification tasks.
    X_train : np.ndarray
        Training input data (observations).
    X_val : np.ndarray
        Validation input data (observations).
    Y_train : np.ndarray
        Training labels (action indices or Q-values).
    Y_val : np.ndarray
        Validation labels (action indices or Q-values).
    all_q : bool, optional
        If True, use regression to predict full Q-values; if False, use classification. Default is False.
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
        max_depth: int = 3,
    ) -> None:
        super().__init__(
            model_activation_obtainer=model_activation_obtainer,
            num_classes=num_classes,
            X_train=X_train,
            X_val=X_val,
            Y_train=Y_train,
            Y_val=Y_val,
            all_q=all_q,
            max_depth=max_depth,
        )

    def _train_ccm_model(
        self,
        concept_scores: List[np.ndarray],
    ) -> DecisionTreeClassifier | DecisionTreeRegressor:
        """
        Train a decision tree to predict model outputs from concept scores.

        Parameters
        ----------
        concept_scores : List[np.ndarray]
            List of concept score vectors for the training data.

        Returns
        -------
        DecisionTreeClassifier or DecisionTreeRegressor
            Trained decision tree model:
            - DecisionTreeClassifier if classification mode is used (`all_q=False`).
            - DecisionTreeRegressor if regression mode is used (`all_q=True`).
        """
        X = np.vstack(concept_scores)

        if self.all_q:
            clf = DecisionTreeRegressor(
                max_depth=self.max_depth, random_state=28, min_samples_leaf=5
            )
        else:
            clf = DecisionTreeClassifier(
                max_depth=self.max_depth, random_state=28, min_samples_leaf=5
            )
        clf.fit(X, self.Y_train)

        return clf
