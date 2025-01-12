from abc import ABC, abstractmethod


class ConceptExampleCollector(ABC):
    """
    Abstract base class for collecting and managing concept examples.

    This class defines the interface for collecting concept examples
    from an environment, managing them, and saving them to disk.
    Subclasses must implement the abstract methods.
    """

    @abstractmethod
    def _env_reset_collect_concept_examples(self, example_n: int) -> None:
        """
        Collect concept examples by resetting the environment.

        This method should be implemented to collect a specific number
        of examples by resetting the environment.

        Parameters
        ----------
        example_n : int
            The number of examples to collect.
        """
        pass

    @abstractmethod
    def collect_examples(self, example_n: int, method: str = "env_reset") -> None:
        """
        Collect concept examples using a specified method.

        Subclasses should implement this method to support different
        collection methods.

        Parameters
        ----------
        example_n : int
            The number of examples to collect.
        method : str, optional
            The method used to collect examples. Default is "env_reset".
        """
        pass

    @abstractmethod
    def save_examples(self, directory_path: str) -> None:
        """
        Save collected concept examples to disk.

        Subclasses should implement this method to save collected examples
        to the specified directory in an appropriate format.

        Parameters
        ----------
        directory_path : str
            The path to the directory where examples will be saved.
        """
        pass
