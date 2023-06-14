# stdlib
from abc import ABCMeta, abstractmethod
from typing import Any, List, Tuple

# third party
from optuna.trial import Trial


class Params(metaclass=ABCMeta):
    def __init__(self, name: str, bounds: Tuple[Any, Any]) -> None:
        self.name = name
        self.bounds = bounds

    @abstractmethod
    def get(self) -> List[Any]:
        ...

    @abstractmethod
    def sample(self, trial: Trial) -> Any:
        ...


class Categorical(Params):
    def __init__(self, name: str, choices: List[Any]) -> None:
        super().__init__(name, (min(choices), max(choices)))
        self.name = name
        self.choices = choices

    def get(self) -> List[Any]:
        return [self.name, self.choices]

    def sample(self, trial: Trial) -> Any:
        return trial.suggest_categorical(self.name, self.choices)


class Float(Params):
    def __init__(self, name: str, low: float, high: float, log: bool = False) -> None:
        low = float(low)
        high = float(high)

        super().__init__(name, (low, high))
        self.name = name
        self.low = low
        self.high = high
        self.log = log

    def get(self) -> List[Any]:
        return [self.name, self.low, self.high, self.log]

    def sample(self, trial: Trial) -> float:
        return trial.suggest_float(self.name, self.low, self.high, log=self.log)

class Integer(Params):
    def __init__(self, name: str, low: int, high: int, step: int = 1) -> None:
        self.low = low
        self.high = high
        self.step = step

        super().__init__(name, (low, high))
        self.name = name
        self.low = low
        self.high = high
        self.step = step

    def get(self) -> List[Any]:
        return [self.name, self.low, self.high, self.step]

    def sample(self, trial: Trial) -> Any:
        return trial.suggest_int(self.name, self.low, self.high, self.step)
