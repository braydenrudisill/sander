from dataclasses import dataclass
from enum import Enum

class JointName(Enum):
    SWIVEL = 0
    LOWER_ARM = 1
    UPPER_ARM = 2
    ROLL = 3
    BEND = 4
    TWIST = 5

type JointState = tuple[float, float, float, float, float, float]
