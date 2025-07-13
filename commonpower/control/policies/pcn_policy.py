import numpy as np
import torch
from gymnasium import Env
from morl_baselines.multi_policy.pcn.pcn import PCN

from commonpower.control.logging_utils.loggers import WandBLoggerSB3
from commonpower.control.policies.base_policy import BasePolicy


class PCNPolicy(BasePolicy):
    """
    PCNPolicy is a policy that uses the PCN (Power Control Network) to control the power consumption of devices.
    It inherits from BasePolicy and implements the necessary methods to interact with the PCN.
    """

    is_morl = True

    def __init__(self, env: Env, seed: int, callback=None, **library_specific_kwargs):
        super().__init__(env, seed, callback)

        self.__dict__.update(library_specific_kwargs)

        self.logger = WandBLoggerSB3()

        constructor_keys = [
            'scaling_factor',
            'learning_rate',
            'gamma',
            'batch_size',
            'hidden_dim',
            'noise',
            'device',
            'model_class',
        ]
        self.constructor_kwargs = {k: v for k, v in library_specific_kwargs.items() if k in constructor_keys}
        self.learn_kwargs = {k: v for k, v in library_specific_kwargs.items() if k not in constructor_keys}

        # Set for no wandb logging during deployment
        self.constructor_kwargs['log'] = False

        self.library_specific_policy = PCN(env=env, seed=seed, **self.constructor_kwargs)
        self.logger = WandBLoggerSB3()

        # Back up original methods
        self._original_run_episode = self.library_specific_policy._run_episode
        self._original_update = self.library_specific_policy.update

        # Apply wrappers
        self._wrap_methods()

    def get_env(self) -> Env:
        return self.library_specific_policy.env

    def load(self, path: str) -> None:
        self.library_specific_policy.model = torch.load(path, map_location=torch.device('cpu'), weights_only=False)

    def save(self, path: str) -> None:
        torch.save(self.library_specific_policy.model, path)

    def learn(self, total_timesteps: int, **kwargs) -> None:
        eval_env = kwargs['eval_env']
        ref_point = kwargs['ref_point']
        self.library_specific_policy.train(total_timesteps=total_timesteps, eval_env=eval_env, ref_point=ref_point)

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.library_specific_policy.eval(obs), None

    def set_deployment_preferences(self, desired_return: np.ndarray, desired_horizon: int):
        """
        Sets the desired return and horizon for the PCN agent during deployment.
        """
        if hasattr(self.library_specific_policy, 'set_desired_return_and_horizon'):
            self.library_specific_policy.set_desired_return_and_horizon(desired_return, desired_horizon)

    def _wrap_methods(self):
        """Wrap PCN methods to inject callback calls."""

        # Create wrapped run_episode method
        def wrapped_run_episode(env, desired_return, desired_horizon, max_return, eval_mode=True):
            transitions = self._original_run_episode(env, desired_return, desired_horizon, max_return, eval_mode)

            # Update num_timesteps
            self.num_timesteps = self.library_specific_policy.global_step

            # Call callbacks
            if self.callback:
                self.callback.on_rollout_end()

            return transitions

        # Create wrapped update method
        def wrapped_update():
            result = self._original_update()

            # Update num_timesteps
            self.num_timesteps = self.library_specific_policy.global_step

            # Call callbacks
            if self.callback:
                self.callback.on_step()

            self.logger.dump(self.num_timesteps)

            return result

        # Apply wrappers
        self.library_specific_policy._run_episode = wrapped_run_episode
        self.library_specific_policy.update = wrapped_update
