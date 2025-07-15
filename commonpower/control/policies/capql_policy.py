import os

import numpy as np
import torch
from gymnasium import Env
from morl_baselines.multi_policy.capql.capql import CAPQL

from commonpower.control.policies.morl_policy import MORLPolicy


class CAPQLPolicy(MORLPolicy):
    """
    Facade for morl-baseline's CAPQL policy.
    """

    def __init__(self, env: Env, seed: int, callback=None, **library_specific_kwargs):
        super().__init__(env, seed, callback)

        init_keys = [
            'learning_rate',
            'batch_size',
            'gamma',
            'tau',
            'buffer_size',
            'net_arch',
            'num_q_nets',
            'alpha',
            'learning_starts',
            'gradient_updates',
            'project_name',
            'experiment_name',
            'wandb_entity',
            'log',
            'device',
        ]
        self.init_kwargs = {k: v for k, v in library_specific_kwargs.items() if k in init_keys}
        self.learn_kwargs = {k: v for k, v in library_specific_kwargs.items() if k not in init_keys}

        self.library_specific_policy = CAPQL(env=env, seed=seed, **self.init_kwargs)

    def load(self, path: str) -> None:
        load_backup = torch.load

        def load_override(path, map_location):
            return load_backup(path, map_location, weights_only=False)

        torch.load = load_override
        self.library_specific_policy.load(path=os.path.join(path, 'model.tar'))
        torch.load = load_backup

    def save(self, path: str) -> None:
        self.library_specific_policy.save(save_dir=path, filename='model')

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self.preference_vector is None:
            raise ValueError('Must call set_prediction_parameters first!')
        return self.library_specific_policy.eval(obs, self.preference_vector), None

    def set_prediction_parameters(self, preference_vector: np.ndarray):
        self.preference_vector = preference_vector
