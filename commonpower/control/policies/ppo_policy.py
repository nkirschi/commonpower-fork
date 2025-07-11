from typing import Self

import numpy as np
from gymnasium import Env
from stable_baselines3 import PPO

from commonpower.control.policies.base_policy import BasePolicy


class PPOPolicy(BasePolicy):
    """
    PPOPolicy is a policy that uses Proximal Policy Optimization (PPO) to control the power consumption of devices.
    It inherits from BasePolicy and implements the necessary methods.
    """

    def __init__(self, env: Env, seed: int, callback=None, **library_specific_kwargs):
        super().__init__(env, seed, callback)
        self.library_specific_policy = PPO(env=env, seed=seed, **library_specific_kwargs)

    def get_env(self) -> Env:
        return self.library_specific_policy.get_env()

    def load(self, path: str) -> None:
        self.library_specific_policy = self.library_specific_policy.load(path)

    def save(self, path: str) -> None:
        self.library_specific_policy.save(path)

    def learn(self, total_timesteps: int, **kwargs) -> Self:
        log_interval = kwargs['log_interval']
        self.library_specific_policy.learn(
            total_timesteps=total_timesteps, callback=self.callback, log_interval=log_interval
        )

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.library_specific_policy.predict(obs, deterministic=deterministic)
