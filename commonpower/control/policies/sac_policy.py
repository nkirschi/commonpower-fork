from gymnasium import Env

from commonpower.control.policies.base_policy import BasePolicy


class SACPolicy(BasePolicy):
    is_morl = False

    def __init__(self, env: Env, seed: int, callback=None, **library_specific_kwargs):
        raise NotImplementedError('SACPolicy is not yet implemented')  # TODO: implement me
