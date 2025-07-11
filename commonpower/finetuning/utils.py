from enum import Enum, auto

from commonpower.control.policies.pcn_policy import PCNPolicy
from commonpower.control.policies.ppo_policy import PPOPolicy


class CEnum(Enum):
    def __str__(self):
        return self.name


class Stage(CEnum):
    Train = auto()
    Deploy = auto()


class Approach(CEnum):
    WithProjectionSafeguard = auto()
    WithReplacementSafeguard = auto()
    OptimalController = auto()


class Penalty(CEnum):
    NoPenalty = auto()
    ConstantPenalty = auto()
    DDPenalty = auto()
    BothPenalties = auto()


class RLAlgorithm(CEnum):
    PPO = auto()
    PCN = auto()

    def is_morl(self):
        return self in [RLAlgorithm.PCN]

    def to_policy_class(self):
        if self == RLAlgorithm.PCN:
            return PCNPolicy
        elif self == RLAlgorithm.PPO:
            return PPOPolicy
        else:
            raise ValueError(f"Unsupported RL algorithm: {self}")
