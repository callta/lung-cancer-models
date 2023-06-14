# stdlib
from typing import Any, Generator, List, Type, Union

# adjutorium absolute
from adjutorium.plugins.prediction.classifiers import Classifiers
from adjutorium.plugins.prediction.risk_estimation import RiskEstimation

# adjutorium relative
from .base import PredictionPlugin  # noqa: F401,E402


class Predictions:
    def __init__(self, category: str = "classifier") -> None:
        self._category = category

        self._plugins: Union[Classifiers, RiskEstimation]

        if category == "classifier":
            self._plugins = Classifiers()
        elif category == "risk_estimation":
            self._plugins = RiskEstimation()
        else:
            raise ValueError(f"unsupported category {category}")

    def list(self) -> List[str]:
        return self._plugins.list()

    def add(self, name: str, cls: Type) -> "Predictions":
        self._plugins.add(name, cls)

        return self

    def get(self, name: str, *args: Any, **kwargs: Any) -> PredictionPlugin:
        return self._plugins.get(name, *args, **kwargs)

    def get_type(self, name: str) -> Type:
        return self._plugins.get_type(name)

    def __iter__(self) -> Generator:
        for x in self._plugins:
            yield x

    def __len__(self) -> int:
        return len(self.list())

    def __getitem__(self, key: str) -> PredictionPlugin:
        return self.get(key)

    def reload(self) -> "Predictions":
        self._plugins = Classifiers()
        return self


__all__ = [
    "Predictions",
    "PredictionPlugin",
]
