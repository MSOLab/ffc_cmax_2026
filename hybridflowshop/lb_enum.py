from enum import Enum, auto


class LbModelType(Enum):
    RELAXED = auto()
    PARALLEL_MC = auto()


class AggregationType(Enum):
    FIRST = auto()
    AVERAGE = auto()
