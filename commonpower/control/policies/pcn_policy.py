import os

import numpy as np
import torch
from gymnasium import Env
from morl_baselines.multi_policy.pcn.pcn import PCN

from commonpower.control.policies.morl_policy import MORLPolicy


class PCNPolicy(MORLPolicy):
    """
    Facade for morl-baseline's PCN policy.
    """

    def __init__(self, env: Env, seed: int, callback=None, **library_specific_kwargs):
        super().__init__(env, seed, callback)

        constructor_keys = [
            'learning_rate',
            'batch_size',
            'gamma',
            'scaling_factor',
            'hidden_dim',
            'noise',
            'device',
            'model_class',
            'log',
            'project_name',
            'wandb_entity',
            'experiment_name',
        ]
        self.init_kwargs = {k: v for k, v in library_specific_kwargs.items() if k in constructor_keys}
        self.train_kwargs = {k: v for k, v in library_specific_kwargs.items() if k not in constructor_keys}

        self.library_specific_policy = PCN(env=env, seed=seed, **self.init_kwargs)

        try:
            desired_return = library_specific_kwargs.pop('desired_return')
            desired_horizon = library_specific_kwargs.pop('desired_horizon')
            self.library_specific_policy.set_desired_return_and_horizon(desired_return, desired_horizon)
        except KeyError:
            print('Info: no desired_return and desired_horizon has been set for PCN yet')

    def load(self, path: str) -> None:
        self.library_specific_policy.model = torch.load(
            os.path.join(path, 'model.zip'), map_location=self.library_specific_policy.device, weights_only=False
        )

    def save(self, path: str) -> None:
        torch.save(self.library_specific_policy.model, os.path.join(path, 'model.zip'))

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.library_specific_policy.eval(obs), None

    def set_prediction_parameters(self, desired_return: np.ndarray, desired_horizon: int):
        self.library_specific_policy.set_desired_return_and_horizon(desired_return, desired_horizon)
