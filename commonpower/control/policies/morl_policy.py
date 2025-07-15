from gymnasium import Env

from commonpower.control.logging_utils.loggers import WandBLoggerSB3
from commonpower.control.policies.base_policy import BasePolicy


class MORLPolicy(BasePolicy):
    """
    Facade for policies from the MORL Baselines library (https://lucasalegre.github.io/morl-baselines/)
    """

    is_morl = True

    def __init__(self, env: Env, seed: int, callback=None, **library_specific_kwargs):
        super().__init__(env, seed, callback)

        self.logger = WandBLoggerSB3()

    def get_env(self) -> Env:
        return self.library_specific_policy.env

    def learn(self, total_timesteps: int, **kwargs) -> None:
        eval_env = kwargs['eval_env']
        ref_point = kwargs['ref_point']
        self._wrap_env_reset()
        self.library_specific_policy.train(
            total_timesteps=total_timesteps, eval_env=eval_env, ref_point=ref_point, **self.learn_kwargs
        )
        self._unwrap_env_reset()

    def _wrap_env_reset(self):
        """Wrap PCN methods to inject callback calls."""

        self._original_reset = self.env.reset

        def wrapped_reset(*, seed=None, options=None):
            result = self._original_reset(seed=seed, options=options)

            if self.callback:
                self.callback.on_rollout_end()
            self.logger.dump(self.library_specific_policy.global_step)

            return result

        self.env.reset = wrapped_reset

    def _unwrap_env_reset(self):
        self.env.reset = self._original_reset
