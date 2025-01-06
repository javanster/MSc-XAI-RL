import os
import random
import time
from typing import Callable, List, Optional, Tuple, cast

import gymnasium as gym
import numpy as np
import tensorflow as tf
from gymnasium import Env
from keras.api.models import Sequential, clone_model
from tqdm import tqdm

import wandb

from .custom_types.experience import Experience
from .custom_types.model_training_config import ModelTrainingConfig
from .util_classes.observation_normalization_callbacks import ObservationNormalizationCallbacks
from .util_classes.replay_buffer import ReplayBuffer
from .util_classes.reward_queue import RewardQueue
from .util_classes.wandb_logger import WandbLogger


def _update_epsilon(epsilon: float, min_epsilon: float, epsilon_decay: float) -> float:
    """
    Updates the epsilon value for epsilon-greedy action selection.

    Epsilon determines the probability of taking a random action versus a greedy action.
    It is gradually decayed until it reaches the minimum epsilon value.

    Parameters
    ----------
    epsilon : float
        The current epsilon value.
    min_epsilon : float
        The minimum allowable epsilon value.
    epsilon_decay : float
        The decay factor applied to epsilon at each step.

    Returns
    -------
    float
        The updated epsilon value.
    """
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay
        epsilon = max(min_epsilon, epsilon)
    return epsilon


class DDQNAgent:
    """
    A Double Deep Q-Network (DDQN) agent for reinforcement learning.

    The DDQN agent utilizes two neural networks (online and target) to stabilize
    Q-value estimation and mitigate overestimation. It supports experience replay,
    epsilon-greedy action selection, and configurable observation normalization.

    Parameters
    ----------
    env : gymnasium.Env
        The environment instance for the agent to interact with. The action space must be discrete.
    obervation_normalization_type : str, optional
        The type of normalization to apply to observations. Defaults to "no_normalization".

    Attributes
    ----------
    env : gymnasium.Env
        The environment instance for the agent.
    normalization_callback : Callable[[np.ndarray], np.ndarray]
        A callback function for normalizing observations based on the selected type.
    best_tumbling_window_average : float
        The highest average reward achieved over a tumbling window during training.
    online_model : keras.api.models.Sequential
        The online neural network model used for predicting Q-values.
    target_model : keras.api.models.Sequential
        The target neural network model used for stabilizing training.
    replay_buffer : ReplayBuffer
        A buffer for storing and sampling past experiences to enable experience replay.
    """

    def __init__(
        self,
        env: Env,
        obervation_normalization_type: str = "no_normalization",
    ) -> None:
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("DDQNAgent only supports discrete action spaces.")

        self.env: Env = env
        self.normalization_callback = self._set_normalization_callback(
            normalization_type=obervation_normalization_type
        )
        self.best_tumbling_window_average: float = float("-inf")

    def _set_normalization_callback(
        self, normalization_type: str
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Sets the normalization callback function based on the specified normalization type.

        The normalization callback is used to preprocess observations before they are passed to the
        neural network. The type of normalization can be selected from predefined options.

        Parameters
        ----------
        normalization_type : str
            The type of normalization to apply. Must be a valid key in
            `ObservationNormalizationCallbacks.normalization_callbacks`.

        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            A function that takes an observation as input and returns the normalized observation.

        Raises
        ------
        ValueError
            If the `normalization_type` is not a valid key in
            `ObservationNormalizationCallbacks.normalization_callbacks`.
        """
        if (
            normalization_type
            not in ObservationNormalizationCallbacks.normalization_callbacks.keys()
        ):
            raise ValueError(
                f'"{normalization_type}" is not a valid argument for normalization_type. Valid arguments: {[type for type in ObservationNormalizationCallbacks.normalization_callbacks.keys()]}'
            )
        return ObservationNormalizationCallbacks.normalization_callbacks[normalization_type]

    def _set_models(self, model: Sequential) -> None:
        """
        Sets the online model and initializes the target model as a clone of the online model.

        Parameters
        ----------
        model : keras.api.models.Sequential
            The neural network model to be used as the online model. The target model is initialized
            as a clone of this model.

        Returns
        -------
        None
        """

        self.online_model: Sequential = model
        self.target_model: Sequential = clone_model(model)

    def _train_network(
        self, min_replay_buffer_size: int, minibatch_size: int, discount: float
    ) -> None:
        """
        Trains the online neural network using a minibatch of experiences from the replay buffer.

        This method samples a random minibatch from the replay buffer, calculates target Q-values
        using the DDQN update rule, and updates the online model. If the replay buffer does
        not contain enough experiences, the method exits early.

        Parameters
        ----------
        min_replay_buffer_size : int
            The minimum number of experiences required in the replay buffer to start training.
        minibatch_size : int
            The number of experiences to sample from the replay buffer for each training step.
        discount : float
            The discount factor (gamma) used to calculate target Q-values.

        Returns
        -------
        None
        """
        if self.replay_buffer.get_size() < min_replay_buffer_size:
            return

        minibatch: List[Experience] = self.replay_buffer.get_random_sample(
            sample_size=minibatch_size
        )

        current_states: np.ndarray = self.normalization_callback(
            np.array([transition[0] for transition in minibatch])
        )
        current_qs_list: np.ndarray = self.online_model.predict(current_states)

        next_states: np.ndarray = self.normalization_callback(
            np.array([transition[3] for transition in minibatch])
        )
        next_qs_list = self.target_model.predict(next_states)

        X = []
        y = []

        for index, (current_state, action, reward, _, terminated) in enumerate(minibatch):
            if not terminated:
                # In accordance with double dqn
                max_next_action = np.argmax(current_qs_list[index])
                max_next_q = next_qs_list[index][max_next_action]
                new_q = reward + discount * max_next_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.online_model.fit(
            x=self.normalization_callback(np.array(X)),
            y=np.array(y),
            batch_size=minibatch_size,
            verbose="0",
            shuffle=False,  # Sampling from the replay buffer has already shuffled the data
        )

    def _ensure_and_get_model_dir_path(self, env_name: str, use_wandb: bool) -> str:
        """
        Ensures the directory structure for saving models exists and returns the directory path.

        This method creates a directory for saving models if it does not already exist. The directory
        name is determined based on the environment name and optionally the Weights & Biases (WandB) run name.

        Parameters
        ----------
        env_name : str
            The name of the environment, used to organize model files.
        use_wandb : bool
            Whether Weights & Biases (WandB) is being used for logging. If True, the WandB run name
            is included in the directory path.

        Returns
        -------
        str
            The full path to the directory where models should be saved.
        """
        if not os.path.exists("models"):
            os.makedirs("models")

        model_dir: str = (
            f"models/{env_name}/{wandb.run.name}"
            if use_wandb
            else f"models/{env_name}/run_{int(time.time())}"
        )

        os.makedirs(model_dir, exist_ok=True)

        return model_dir

    def _save_training_info(self, directory_path: str, config: ModelTrainingConfig) -> None:
        """
        Saves detailed information about the training configuration and model architecture.

        This method writes the architecture of the online model and the training configuration
        to a text file named `info.txt` in the specified directory. The information includes
        layer details and the key-value pairs from the provided configuration.

        Parameters
        ----------
        directory_path : str
            The path to the directory where the `info.txt` file should be saved.
        config : ModelTrainingConfig
            A dictionary containing the training configuration settings.

        Returns
        -------
        None
        """
        with open(f"{directory_path}/info.txt", "w") as file:
            file.write("Detailed Model Summary\n")
            file.write("=" * 50 + "\n")
            for i, layer in enumerate(self.online_model.layers):
                file.write(f"Layer {i + 1}: {layer.name}\n")
                file.write(f"  Type: {layer.__class__.__name__}\n")
                layer_config = layer.get_config()
                for key, value in layer_config.items():
                    file.write(f"  {key}: {value}\n")
                file.write("-" * 50 + "\n")

            file.write("\n\nTraining Config\n")
            file.write("=" * 50 + "\n")
            for key, value in config.items():
                file.write(f"{key}: {value}\n")

    def _choose_action_epsilon_greedy(self, epsilon: float, current_state: np.ndarray) -> int:
        """
        Selects an action using the epsilon-greedy strategy, with support for negative epsilon.

        With probability `epsilon`, a random action is chosen (exploration). Otherwise, the action
        with the highest predicted Q-value is selected (exploitation) based on the current state.
        A negative epsilon ensures that a random action is never taken, forcing exploitation only.

        Parameters
        ----------
        epsilon : float
            The probability of choosing a random action. Must be in the range [-inf, 1].
            Negative epsilon ensures pure exploitation, where the best action is always chosen.
        current_state : np.ndarray
            The current state of the environment, used as input to the online model for predicting Q-values.

        Returns
        -------
        int
            The selected action, represented as an integer.

        Raises
        ------
        ValueError
            If `epsilon` is greater than 1.
        """
        if epsilon > 1:
            raise ValueError("Given epsilon must be within range [-inf, 1]")

        if np.random.random() > epsilon:
            state_reshaped = self.normalization_callback(
                np.array(current_state).reshape(-1, *current_state.shape)
            )
            q_values = self.online_model.predict(state_reshaped)[0]
            action = int(np.argmax(q_values))
        else:
            action = int(np.random.randint(0, self.env.action_space.n))

        return action

    def _update_models(
        self,
        steps_passed: int,
        training_frequency: int,
        update_target_every: int,
        min_replay_buffer_size: int,
        minibatch_size: int,
        discount: float,
    ) -> None:
        """
        Updates the online and target models during training based on the current step count.

        This method triggers the training of the online model at regular intervals defined by
        `training_frequency` and updates the target model's weights to match the online model's
        weights at intervals defined by `update_target_every`.

        Parameters
        ----------
        steps_passed : int
            The number of steps that have been executed in the training process.
        training_frequency : int
            The number of steps between training updates for the online model.
        update_target_every : int
            The number of steps between updates of the target model's weights.
        min_replay_buffer_size : int
            The minimum number of experiences required in the replay buffer to perform training.
        minibatch_size : int
            The number of experiences to sample from the replay buffer for each training update.
        discount : float
            The discount factor (gamma) used in the Q-value update calculations.

        Returns
        -------
        None
        """
        if steps_passed % training_frequency == 0:
            self._train_network(min_replay_buffer_size, minibatch_size, discount)

        if steps_passed % update_target_every == 0:
            self.target_model.set_weights(self.online_model.get_weights())

    def _reset_env_and_observe(self) -> Tuple[np.ndarray, float, bool, bool]:
        """
        Resets the environment and initializes episode-related variables.

        This method resets the environment to its initial state and prepares the variables
        required to track the progress of an episode, such as the initial state, cumulative
        reward, and termination status.

        Returns
        -------
        Tuple[np.ndarray, float, bool, bool]
            A tuple containing:
            - initial_state (np.ndarray): The initial state of the environment after reset.
            - episode_reward (float): The cumulative reward for the episode, initialized to 0.
            - terminated (bool): Whether the episode has been terminated, initialized to False.
            - truncated (bool): Whether the episode has been truncated, initialized to False.
        """
        initial_state, _ = self.env.reset()
        initial_state = cast(np.ndarray, initial_state)
        episode_reward: float = 0
        terminated: bool = False
        truncated: bool = False
        return initial_state, episode_reward, terminated, truncated

    def _perform_env_step_and_observe(self, action: int) -> Tuple[np.ndarray, float, bool, bool]:
        """
        Executes a step in the environment using the specified action and observes the results.

        This method sends an action to the environment, retrieves the resulting state, reward,
        and episode status, and casts the values to their appropriate types.

        Parameters
        ----------
        action : int
            The action to be performed in the environment.

        Returns
        -------
        Tuple[np.ndarray, float, bool, bool]
            A tuple containing:
            - new_state (np.ndarray): The state of the environment after the action is taken.
            - reward (float): The reward received from the environment for the action.
            - terminated (bool): Whether the episode has been terminated.
            - truncated (bool): Whether the episode has been truncated.
        """
        new_state, reward, terminated, truncated, _ = self.env.step(action=action)
        new_state = cast(np.ndarray, new_state)
        reward = cast(float, reward)
        terminated = cast(bool, terminated)
        truncated = cast(bool, truncated)
        return new_state, reward, terminated, truncated

    def _save_online_model(
        self, model_dir: str, average_reward: float, max_reward: float, min_reward: float
    ) -> None:
        """
        Saves the online model to a file with a name that includes performance metrics.

        This method saves the online model to the specified directory. The filename
        includes a timestamp and performance metrics (average reward, maximum reward,
        and minimum reward) to uniquely identify the saved model.

        Parameters
        ----------
        model_dir : str
            The directory where the model file will be saved.
        average_reward : float
            The average reward achieved during a training period, included in the filename.
        max_reward : float
            The maximum reward achieved during a training period, included in the filename.
        min_reward : float
            The minimum reward achieved during a training period, included in the filename.

        Returns
        -------
        None
        """
        model_file_path = f"{model_dir}/{int(time.time())}_model_{average_reward:_>9.4f}avg_{max_reward:_>9.4f}max_{min_reward:_>9.4f}min.keras"
        self.online_model.save(model_file_path)

    def train(
        self,
        config: ModelTrainingConfig,
        model: Sequential,
        use_wandb: bool = False,
        use_sweep: bool = False,
    ) -> None:
        """
        Trains the DDQN agent in the specified environment.

        This method trains the agent using the Double Deep Q-Network (DDQN) algorithm.
        It initializes the necessary components, including the replay buffer, reward tracking,
        and epsilon decay schedule. The training process is logged step-by-step and episode-by-episode,
        with optional integration into Weights & Biases (WandB) for detailed tracking.

        Parameters
        ----------
        config : ModelTrainingConfig
            A dictionary containing training configuration parameters. Must include:
            - `env_name`: Name of the environment.
            - `replay_buffer_size`: Maximum size of the replay buffer.
            - `min_replay_buffer_size`: Minimum size of the replay buffer before training starts.
            - `minibatch_size`: Number of experiences to sample for each training step.
            - `discount`: Discount factor for Q-value updates.
            - `training_frequency`: Steps between training updates.
            - `update_target_every`: Steps between target model updates.
            - `steps_to_train`: Total number of training steps.
            - `starting_epsilon`: Initial epsilon value for epsilon-greedy exploration.
            - `prop_steps_epsilon_decay`: Proportion of total steps over which epsilon decays.
            - `min_epsilon`: Minimum allowable epsilon value.
            - `episode_metrics_window`: Tumbling window size for reward metrics.
        model : keras.api.models.Sequential
            A pre-defined model to use for both the online and target networks.
        use_wandb : bool, optional
            Whether to log training progress and metrics to Weights & Biases (WandB). Defaults to False.
        use_sweep : bool, optional
            Whether the training is part of a WandB sweep (hyperparameter tuning). Defaults to False.

        Returns
        -------
        None
        """

        wandbLogger = WandbLogger(log_active=use_wandb, sweep_active=use_sweep, config=config)
        if use_wandb:
            config = wandb.config

        env_name: str = config["env_name"]
        replay_buffer_size: int = config["replay_buffer_size"]
        min_replay_buffer_size: int = config["min_replay_buffer_size"]
        minibatch_size: int = config["minibatch_size"]
        discount: float = config["discount"]
        training_frequency: int = config["training_frequency"]
        update_target_every: int = config["update_target_every"]
        steps_to_train: int = config["steps_to_train"]
        epsilon: float = config["starting_epsilon"]
        starting_epsilon: float = config["starting_epsilon"]
        prop_steps_epsilon_decay: float = config["prop_steps_epsilon_decay"]
        min_epsilon: float = config["min_epsilon"]
        episode_metrics_window: int = config["episode_metrics_window"]

        self.replay_buffer: ReplayBuffer = ReplayBuffer(maxlen=replay_buffer_size)
        reward_queue: RewardQueue = RewardQueue(maxlen=episode_metrics_window)
        num_decay_steps: float = steps_to_train * prop_steps_epsilon_decay
        epsilon_decay: float = (min_epsilon / starting_epsilon) ** (1 / num_decay_steps)
        steps_passed: int = 0
        episodes_passed: int = 0
        average_reward = None
        max_reward = None
        min_reward = None
        model_dir = self._ensure_and_get_model_dir_path(env_name=env_name, use_wandb=use_wandb)

        self._set_models(
            model=model,
        )

        self._save_training_info(model_dir, config)

        random.seed(28)
        np.random.seed(28)
        tf.random.set_seed(28)

        # TRAIN LOOP, terminates when given training steps have passed
        with tqdm(total=steps_to_train, unit="step") as pbar:
            while steps_passed < steps_to_train:

                current_state, episode_reward, terminated, truncated = self._reset_env_and_observe()

                # STEP LOOP, terminates when an episode ends
                while not terminated and not truncated:
                    action: int = self._choose_action_epsilon_greedy(
                        epsilon=epsilon, current_state=current_state
                    )

                    new_state, reward, terminated, truncated = self._perform_env_step_and_observe(
                        action
                    )

                    steps_passed += 1
                    episode_reward += reward

                    self.replay_buffer.update(
                        Experience(current_state, action, reward, new_state, terminated)
                    )

                    self._update_models(
                        steps_passed=steps_passed,
                        training_frequency=training_frequency,
                        update_target_every=update_target_every,
                        min_replay_buffer_size=min_replay_buffer_size,
                        minibatch_size=minibatch_size,
                        discount=discount,
                    )

                    wandbLogger.log_step_data(steps_passed=steps_passed, epsilon=epsilon)

                    current_state = new_state

                    epsilon = _update_epsilon(
                        epsilon=epsilon, min_epsilon=min_epsilon, epsilon_decay=epsilon_decay
                    )

                    pbar.update(1)
                    if steps_passed >= steps_to_train:
                        break

                episodes_passed += 1
                reward_queue.update(episode_reward)

                wandbLogger.log_episode_data(
                    episodes_passed=episodes_passed,
                    reward_queue=reward_queue,
                    episode_reward=episode_reward,
                    episode_metrics_window=episode_metrics_window,
                )

                if (
                    reward_queue.get_size() >= episode_metrics_window
                    and episodes_passed % episode_metrics_window == 0
                ):
                    average_reward = reward_queue.get_average_reward()

                    # Checks every tumbling window episode whether a new best static average is reached
                    if average_reward > self.best_tumbling_window_average:
                        self.best_tumbling_window_average = average_reward

                        max_reward = reward_queue.get_max_reward()
                        min_reward = reward_queue.get_min_reward()

                        self._save_online_model(
                            model_dir=model_dir,
                            average_reward=average_reward,
                            max_reward=max_reward,
                            min_reward=min_reward,
                        )

            self.env.close()

        # Save final model
        if average_reward and max_reward and min_reward:
            self._save_online_model(
                model_dir=model_dir,
                average_reward=average_reward,
                max_reward=max_reward,
                min_reward=min_reward,
            )

    def test(
        self, model: Sequential, episodes: int = 10, env: Optional[Env] = None
    ) -> Tuple[List[float], int, int]:
        """
        Tests the trained DDQN agent in the environment.

        This method evaluates the agent's performance using a trained model over a specified
        number of episodes. It tracks episode rewards, terminations, and truncations. If the
        environment is configured to render, the testing process can be visualized.

        Parameters
        ----------
        model : keras.api.models.Sequential
            The trained model to be used for evaluation.
        episodes : int, optional
            The number of episodes to run during testing. Defaults to 10.
        env : Optional[gymnasium.Env], optional
            An optional environment instance to override the agent's current environment.
            If not provided, the agent's existing environment is used. Defaults to None.

        Returns
        -------
        Tuple[List[float], int, int]
            A tuple containing:
            - episode_rewards (List[float]): A list of total rewards achieved in each episode.
            - terminations (int): The total number of episodes that ended with termination.
            - truncations (int): The total number of episodes that ended with truncation.

        Notes
        -----
        - The environment's render mode must be set to "human" for visualization.
        """

        if env:
            self.env = env

        render: bool = False
        if self.env and self.env.unwrapped.render_mode == "human":
            render = True

        self._set_models(model=model)

        episode_rewards: List[float] = []
        terminations: int = 0
        truncations: int = 0

        with tqdm(total=episodes, unit="Episode") as pbar:
            for _ in range(episodes):

                observation, episode_reward, terminated, truncated = self._reset_env_and_observe()

                if render:
                    self.env.render()

                while not terminated and not truncated:
                    action = self._choose_action_epsilon_greedy(
                        epsilon=-1,  # Negative epsilon ensures that a random action is never taken
                        current_state=observation,
                    )

                    observation, reward, terminated, truncated = self._perform_env_step_and_observe(
                        action
                    )

                    episode_reward += reward

                    if render:
                        self.env.render()

                if render:
                    time.sleep(2)

                episode_rewards.append(episode_reward)
                if terminated:
                    terminations += 1
                if truncated:
                    truncations += 1

                pbar.update(1)

        self.env.close()

        return episode_rewards, terminations, truncations
