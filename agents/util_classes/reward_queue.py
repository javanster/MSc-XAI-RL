from collections import deque
from typing import Deque


class RewardQueue:
    """
    A class to manage a queue of rewards with a fixed maximum length.

    The `RewardQueue` maintains a fixed-size queue of rewards. When a new reward is added
    and the queue is full, the oldest reward is automatically discarded. It provides
    functionality to calculate statistics such as the average, maximum, and minimum of the rewards
    currently in the queue.

    Attributes
    ----------
    queue : collections.deque
        A deque containing the stored rewards, with a maximum length specified during initialization.

    Methods
    -------
    __init__(maxlen: int) -> None
        Initializes the reward queue with a specified maximum size.
    update(reward: float) -> None
        Adds a new reward to the queue.
    get_size() -> int
        Returns the current number of rewards in the queue.
    get_average_reward() -> float
        Returns the average reward of all rewards in the queue.
    get_max_reward() -> float
        Returns the maximum reward in the queue.
    get_min_reward() -> float
        Returns the minimum reward in the queue.
    """

    def __init__(self, maxlen: int) -> None:
        self.queue: Deque[float] = deque(maxlen=maxlen)

    def update(self, reward: float) -> None:
        """
        Adds a new reward to the reward queue.

        If the queue is full, the oldest reward will be discarded.

        Parameters
        ----------
        reward : float
            The reward to add to the queue.
        """
        self.queue.append(reward)

    def get_size(self) -> int:
        """
        Returns the current number of rewards stored in the queue.

        Returns
        -------
        int
            The number of rewards in the queue.
        """
        return len(self.queue)

    def get_average_reward(self) -> float:
        """
        Returns the average reward of all rewards currently stored in the queue.

        Returns
        -------
        float
            The average reward of rewards in the queue.

        Raises
        ------
        ValueError
            If the queue is empty.
        """
        if self.get_size() == 0:
            raise ValueError("Cannot compute the average of an empty queue.")
        return sum(self.queue) / self.get_size()

    def get_max_reward(self) -> float:
        """
        Returns the maximum reward currently stored in the queue.

        Returns
        -------
        float
            The maximum reward in the queue.

        Raises
        ------
        ValueError
            If the queue is empty.
        """
        if not self.queue:
            raise ValueError("The queue is empty.")
        return max(self.queue)

    def get_min_reward(self) -> float:
        """
        Returns the minimum reward currently stored in the queue.

        Returns
        -------
        float
            The minimum reward in the queue.

        Raises
        ------
        ValueError
            If the queue is empty.
        """
        if not self.queue:
            raise ValueError("The queue is empty.")
        return min(self.queue)
