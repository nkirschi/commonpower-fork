from enum import Enum, auto

from commonpower.control.policies.pcn_policy import PCNPolicy
from commonpower.control.policies.ppo_policy import PPOPolicy
from commonpower.control.policies.sac_policy import SACPolicy


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
    SAC = auto()
    PCN = auto()

    def to_policy_class(self):
        match self:
            case RLAlgorithm.PPO:
                return PPOPolicy
            case RLAlgorithm.SAC:
                return SACPolicy
            case RLAlgorithm.PCN:
                return PCNPolicy
            case _:
                raise ValueError(f"Unsupported RL algorithm: {self}")
