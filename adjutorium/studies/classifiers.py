# stdlib
from pathlib import Path
import time
from typing import Any, List, Optional, Tuple

# third party
import pandas as pd

# adjutorium absolute
from adjutorium.exceptions import StudyCancelled
from adjutorium.explorers.classifiers_combos import EnsembleSeeker
from adjutorium.explorers.core.defaults import (
    default_classifiers_names,
    default_feature_scaling_names,
)
from adjutorium.hooks import Hooks
import adjutorium.logger as log
from adjutorium.studies._base import DefaultHooks, Study
from adjutorium.studies._preprocessing import dataframe_hash, dataframe_preprocess
from adjutorium.utils.serialization import load_model_from_file, save_model_to_file
from adjutorium.utils.tester import evaluate_estimator

PATIENCE = 10
SCORE_THRESHOLD = 0.65


class ClassifierStudy(Study):
    def __init__(
        self,
        dataset: pd.DataFrame,
        target: str,
        num_iter: int = 20,
        num_ensemble_iter: int = 15,
        num_study_iter: int = 5,
        timeout: int = 360,
        metric: str = "aucroc",
        study_name: Optional[str] = None,
        ensemble_size: int = 3,
        feature_scaling: List[str] = default_feature_scaling_names,
        classifiers: List[str] = default_classifiers_names,
        imputers: List[str] = ["ice"],
        workspace: Path = Path("/tmp"),
        hooks: Hooks = DefaultHooks(),
        score_threshold: float = SCORE_THRESHOLD,
    ) -> None:
        super().__init__()

        self.hooks = hooks
        dataset = pd.DataFrame(dataset)

        imputation_method: Optional[str] = None
        if dataset.isnull().values.any():
            assert len(imputers) > 0, "Please provide at least one imputation method"
            if len(imputers) == 1:
                imputation_method = imputers[0]
                imputers = []
        else:
            imputers = []

        #self.X, _, self.Y, _, _ = dataframe_preprocess(
        #    dataset, target, imputation_method=imputation_method
        #)

        self.X = dataset.drop(columns=[target])
        self.Y = dataset[target]

        self.internal_name = dataframe_hash(dataset)
        self.study_name = study_name if study_name is not None else self.internal_name

        self.output_folder = Path(workspace) / self.study_name
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.output_file = self.output_folder / "model.p"

        self.num_study_iter = num_study_iter

        self.metric = metric
        self.score_threshold = score_threshold

        self.seeker = EnsembleSeeker(
            self.internal_name,
            num_iter=num_iter,
            num_ensemble_iter=num_ensemble_iter,
            timeout=timeout,
            metric=metric,
            ensemble_size=ensemble_size,
            feature_scaling=feature_scaling,
            classifiers=classifiers,
            imputers=imputers,
            hooks=self.hooks,
        )

    def _should_continue(self) -> None:
        if self.hooks.cancel():
            raise StudyCancelled("Classifier study search cancelled")

    def load_progress(self) -> Tuple[int, Any]:
        self._should_continue()

        if not self.output_file.is_file():
            return -1, None

        try:
            start = time.time()
            best_model = load_model_from_file(self.output_file)
            metrics = evaluate_estimator(best_model, self.X, self.Y, metric=self.metric)
            best_score = metrics["clf"][self.metric][0]
            self.hooks.heartbeat(
                topic="classification_study",
                subtopic="candidate",
                event_type="candidate",
                name=best_model.name(),
                models=[mod.name() for mod in best_model.models],
                weights=best_model.weights,
                duration=time.time() - start,
                aucroc=metrics["str"]["aucroc"],
            )

            return best_score, best_model
        except BaseException:
            return -1, None

    def save_progress(self, model: Any) -> None:
        self._should_continue()

        if self.output_file:
            save_model_to_file(self.output_file, model)

    def run(self) -> Any:
        self._should_continue()

        best_score, best_model = self.load_progress()

        patience = 0
        for it in range(self.num_study_iter):
            self._should_continue()
            start = time.time()

            current_model = self.seeker.search(self.X, self.Y)

            metrics = evaluate_estimator(
                current_model, self.X, self.Y, metric=self.metric
            )
            score = metrics["clf"][self.metric][0]

            self.hooks.heartbeat(
                topic="classification_study",
                subtopic="candidate",
                event_type="candidate",
                name=current_model.name(),
                duration=time.time() - start,
                aucroc=metrics["str"][self.metric],
            )

            if score < self.score_threshold:
                log.info(f"The ensemble is not good enough, keep searching {metrics}")
                continue

            if best_score >= score:
                log.info(
                    f"Model score not improved {score}. Previous best {best_score}"
                )
                patience += 1

                if patience > PATIENCE:
                    log.info(
                        f"Study not improved for {PATIENCE} iterations. Stopping..."
                    )
                    break
                continue

            patience = 0
            best_score = metrics["clf"][self.metric][0]
            best_model = current_model

            log.error(
                f"Best ensemble so far: {best_model.name()} with score {metrics['clf'][self.metric]}"
            )

            self.save_progress(best_model)

        return best_model
