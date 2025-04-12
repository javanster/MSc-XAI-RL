from typing import List

import numpy as np
from keras.api.models import Sequential
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from utils import ModelActivationObtainer

from .ccm import CCM


class CCM_DT(CCM):

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
        super().__init__(
            model=model,
            model_activation_obtainer=model_activation_obtainer,
            num_classes=num_classes,
            X_train=X_train,
            X_val=X_val,
            Y_train=Y_train,
            Y_val=Y_val,
            all_q=all_q,
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
            - DecisionTreeClassifier if classification mode is used (all_q=False).
            - DecisionTreeRegressor if regression mode is used (all_q=True).
        """
        X = np.vstack(concept_scores)
        if self.all_q:
            clf = DecisionTreeRegressor(max_depth=3, random_state=28)
        else:
            clf = DecisionTreeClassifier(max_depth=3, random_state=28)
        clf.fit(X, self.Y_train)

        return clf
