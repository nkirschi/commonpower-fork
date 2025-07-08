from enum import Enum, auto


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

    def to_algorithm_class(self):
        if self == RLAlgorithm.PCN:
            from morl_baselines.multi_policy.pcn.pcn import PCN

            return PCN
        elif self == RLAlgorithm.PPO:
            from stable_baselines3 import PPO

            return PPO
        else:
            raise ValueError(f"Unsupported RL algorithm: {self}")
