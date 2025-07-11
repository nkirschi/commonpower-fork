from abc import ABC, abstractmethod

import numpy as np
from gymnasium import Env


class BasePolicy(ABC):
    @abstractmethod
    def __init__(self, env: Env, seed: int, callback=None, **library_specific_kwargs):
        """
        Initialize the policy with a specific library's policy and an optional callback.

        Args:
            library_specific_policy: The policy instance from a specific RL library
                                     (currently: Stable Baselines 3 or MORL-Baslines).
            callback (optional): Callback to be used during training.
        """
        self.callback = callback

        if self.callback:
            callback.init_callback(self)

    @abstractmethod
    def get_env(self) -> Env:
        """
        Returns the current environment.

        Returns:
            The current environment the policy operates on
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the policy from a zip-file.

        Args:
            path (str): Path to the file (or a file-like) where to load the agent from

        Returns:
            New model instance with loaded parameters
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the policy parameters in a zip-file.

        Args:
            path (str): Path to the file where the policy should be saved
        """

    @abstractmethod
    def learn(self, total_timesteps: int, **kwargs) -> None:
        """
        Train the policy.

        Args:
            total_timesteps (int): The number of timesteps to train the policy for.
            **kwargs: Additional keyword arguments that may be required by the specific policy implementation.

        Returns:
            The trained model
        """

    @abstractmethod
    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Predict next action given an observation.

        Args:
            obs (np.ndarray): The input observation
            deterministic (bool, optional): Whether or not to return deterministic actions.

        Returns:
            The policy's action and the next hidden state (used in recurrent policies)
        """
