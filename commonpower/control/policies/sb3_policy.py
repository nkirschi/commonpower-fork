import os

import numpy as np
from gymnasium import Env

from commonpower.control.policies.base_policy import BasePolicy


class SB3Policy(BasePolicy):
    """
    Facade for policies from the Stable Baselines 3 library (https://stable-baselines3.readthedocs.io/)
    """

    is_morl = False

    def __init__(self, env: Env, seed: int, callback=None, **library_specific_kwargs):
        super().__init__(env, seed, callback)

    def get_env(self) -> Env:
        return self.library_specific_policy.get_env()

    def load(self, path: str) -> None:
        self.library_specific_policy = self.library_specific_policy.load(os.join(path, 'model.zip'))

    def save(self, path: str) -> None:
        self.library_specific_policy.save(os.join(path, 'model.zip'))

    def learn(self, total_timesteps: int, **kwargs) -> None:
        log_interval = kwargs['log_interval']
        self.library_specific_policy.learn(
            total_timesteps=total_timesteps, callback=self.callback, log_interval=log_interval
        )

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.library_specific_policy.predict(obs, deterministic=deterministic)
