import random
from collections import deque
from typing import Deque, List

from .experience import Experience


class ReplayBuffer:
    """
    A class representing a replay buffer for storing and sampling experiences in reinforcement learning.

    The replay buffer maintains a fixed-size queue of experiences, allowing for efficient
    storage and retrieval of past experiences for training machine learning models.

    Attributes
    ----------
    queue : collections.deque
        A deque containing the stored experiences, with a maximum length specified during initialization.

    Methods
    -------
    __init__(maxlen: int) -> None
        Initializes the replay buffer with a specified maximum size.

    update(experience: Experience) -> None
        Adds a new experience to the replay buffer.

    get_size() -> int
        Returns the current number of experiences stored in the buffer.

    get_random_sample(sample_size: int) -> List[Experience]
        Retrieves a random sample of experiences from the buffer.
    """

    def __init__(self, maxlen: int) -> None:
        self.queue: Deque[Experience] = deque(maxlen=maxlen)

    def update(self, experience: Experience) -> None:
        """
        Adds a new experience to the replay buffer.

        If the buffer is full, the oldest experience will be discarded.

        Parameters
        ----------
        experience : Experience
            The experience to add to the buffer.
        """
        self.queue.append(experience)

    def get_size(self) -> int:
        """
        Returns the current number of experiences stored in the buffer.

        Returns
        -------
        int
            The number of experiences in the buffer.
        """
        return len(self.queue)

    def get_random_sample(self, sample_size: int) -> List[Experience]:
        """
        Retrieves a random sample of experiences from the buffer.

        Parameters
        ----------
        sample_size : int
            The number of experiences to sample from the buffer.

        Returns
        -------
        List[Experience]
            A list containing the sampled experiences.

        Raises
        ------
        ValueError
            If the sample size is greater than the number of experiences in the buffer.
        """
        if self.get_size() < sample_size:
            raise ValueError(
                f"Given sample size of {sample_size} is greater than the replay buffer size of {self.get_size()}"
            )
        return random.sample(self.queue, sample_size)
