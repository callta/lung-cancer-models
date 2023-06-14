# stdlib
from pathlib import Path
import time
from typing import Any, List, Optional, Tuple, Union

# third party
import pandas as pd

# adjutorium absolute
from adjutorium.exceptions import StudyCancelled
from adjutorium.explorers.core.defaults import (
    default_feature_scaling_names,
    default_risk_estimation_names,
)
from adjutorium.explorers.risk_estimation_combos import (
    RiskEnsembleSeeker as standard_seeker,
)
from adjutorium.hooks import Hooks
import adjutorium.logger as log
from adjutorium.studies._base import DefaultHooks, Study
from adjutorium.studies._preprocessing import dataframe_hash, dataframe_preprocess
from adjutorium.utils.serialization import load_model_from_file, save_model_to_file
from adjutorium.utils.tester import evaluate_survival_estimator

PATIENCE = 10
SCORE_THRESHOLD = 0.65


class RiskEstimationStudy(Study):
    def __init__(
        self,
        dataset: pd.DataFrame,
        target: str,
        time_to_event: str,
        time_horizons: List[int],
        num_iter: int = 20,
        num_study_iter: int = 5,
        timeout: int = 360,
        study_name: Optional[str] = None,
        workspace: Path = Path("/tmp"),
        ensemble_size: int = 3,
        risk_estimators: List[str] = default_risk_estimation_names,
        imputers: List[str] = ["ice"],
        feature_scaling: List[str] = default_feature_scaling_names,
        hooks: Hooks = DefaultHooks(),
        score_threshold: float = SCORE_THRESHOLD,
    ) -> None:
        super().__init__()

        # If only one imputation method is provided, we don't feed it into the optimizer
        imputation_method: Optional[str] = None
        if dataset.isnull().values.any():
            assert len(imputers) > 0, "Please provide at least one imputation method"
            if len(imputers) == 1:
                imputation_method = imputers[0]
                imputers = []
        else:
            imputers = []

        self.time_horizons = time_horizons
        self.score_threshold = score_threshold

        #self.X, self.T, self.Y, _, _ = dataframe_preprocess(
        #    dataset,
        #    target,
        #    time_to_event=time_to_event,
        #    imputation_method=imputation_method,
        #)

        self.X = dataset.drop(columns=[target, time_to_event])
        self.Y = dataset[target]
        self.T = dataset[time_to_event]

        self.internal_name = dataframe_hash(dataset)
        self.study_name = study_name if study_name is not None else self.internal_name

        self.output_folder = Path(workspace) / self.study_name
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.output_file = self.output_folder / "model.p"

        self.num_iter = num_iter
        self.num_study_iter = num_study_iter
        self.hooks = hooks

        self.standard_seeker = standard_seeker(
            self.internal_name,
            time_horizons,
            num_iter=num_iter,
            num_ensemble_iter=num_iter,
            timeout=timeout,
            top_k=ensemble_size,
            estimators=risk_estimators,
            feature_scaling=feature_scaling,
            imputers=imputers,
            hooks=hooks,
        )

    def _should_continue(self) -> None:
        if self.hooks.cancel():
            raise StudyCancelled("Risk estimation study search cancelled")

    def _load_progress(self) -> Tuple[int, Any]:
        self._should_continue()

        if not self.output_file.is_file():
            return -1, None

        try:
            log.info("evaluate previous model")
            start = time.time()
            best_model = load_model_from_file(self.output_file)
            metrics = evaluate_survival_estimator(
                best_model, self.X, self.T, self.Y, self.time_horizons
            )
            best_score = metrics["clf"]["c_index"][0] - metrics["clf"]["brier_score"][0]
            self.hooks.heartbeat(
                topic="risk_estimation_study",
                subtopic="candidate",
                event_type="candidate",
                name=best_model.name(),
                models=[mod.name() for mod in best_model.models],
                weights=best_model.weights,
                duration=time.time() - start,
                aucroc=metrics["str"]["aucroc"],
                cindex=metrics["str"]["c_index"],
                brier_score=metrics["str"]["brier_score"],
            )
            log.error(f"Previous best score {best_score}")
            return best_score, best_model
        except BaseException as e:
            log.error(f"failed to load previous model {e}")
            return -1, None

    def _save_progress(self, model: Any) -> None:
        self._should_continue()

        if self.output_file:
            save_model_to_file(self.output_file, model)

    def run(self) -> Any:
        self._should_continue()

        best_score, best_model = self._load_progress()

        seekers: List[Union[standard_seeker]] = [
            self.standard_seeker,
        ]
        not_improved = 0

        for it in range(self.num_study_iter):
            for seeker in seekers:
                self._should_continue()
                start = time.time()

                current_model = seeker.search(
                    self.X,
                    self.T,
                    self.Y,
                    skip_recap=(it > 0),
                )

                metrics = evaluate_survival_estimator(
                    current_model,
                    self.X,
                    self.T,
                    self.Y,
                    self.time_horizons,
                )
                score = metrics["clf"]["c_index"][0] - metrics["clf"]["brier_score"][0]
                self.hooks.heartbeat(
                    topic="risk_estimation_study",
                    subtopic="candidate",
                    event_type="candidate",
                    name=current_model.name(),
                    models=[mod.name() for mod in current_model.models],
                    weights=current_model.weights,
                    duration=time.time() - start,
                    aucroc=metrics["str"]["aucroc"],
                    cindex=metrics["str"]["c_index"],
                    brier_score=metrics["str"]["brier_score"],
                )

                if score < self.score_threshold:
                    log.info(
                        f"The ensemble is not good enough, keep searching {metrics}"
                    )
                    continue

                if best_score >= score:
                    log.info(
                        f"Model score not improved {score}. Previous best {best_score}"
                    )
                    not_improved += 1
                    continue

                not_improved = 0
                best_score = score
                best_model = current_model

                log.error(
                    f"Best ensemble so far: {best_model.name()} with score {score}"
                )

                self._save_progress(best_model)

            if not_improved > PATIENCE:
                log.info(f"Study not improved for {PATIENCE} iterations. Stopping...")
                break

        return best_model
