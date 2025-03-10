import os
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


class TcavExampleCollector(ABC):

    def __init__(self, output_classes_n: int) -> None:
        self.collected_examples: Dict[int, List[np.ndarray]] = {
            action: [] for action in range(output_classes_n)
        }

    @abstractmethod
    def collect_examples(self, examples_per_output_class: int) -> None:
        pass

    @abstractmethod
    def save_examples(self, directory_path: str) -> None:
        pass

    def _ensure_save_directory_exists(self, directory_path: str) -> None:
        """
        Ensure that the save directory exists.

        If the specified directory does not exist, it will be created.

        Parameters
        ----------
        directory_path : str
            The path to the directory to check or create.
        """
        directory = os.path.dirname(directory_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
