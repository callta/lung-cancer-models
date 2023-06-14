# stdlib
import time
import traceback
from typing import Any, Dict, List, Tuple

# third party
from joblib import Parallel, delayed
import numpy as np
import optuna
import pandas as pd

# adjutorium absolute
from adjutorium.exceptions import StudyCancelled
from adjutorium.explorers.core.defaults import (
    default_feature_scaling_names,
    default_risk_estimation_names,
)
from adjutorium.explorers.core.optimizer import EarlyStoppingExceeded, create_study
from adjutorium.explorers.core.selector import PipelineSelector
from adjutorium.explorers.hooks import DefaultHooks
from adjutorium.hooks import Hooks
import adjutorium.logger as log
from adjutorium.utils.parallel import cpu_count
from adjutorium.utils.tester import evaluate_survival_estimator

dispatcher = Parallel(n_jobs=cpu_count())


class RiskEstimatorSeeker:
    def __init__(
        self,
        study_name: str,
        time_horizons: List[int],
        num_iter: int = 50,
        timeout: int = 360,
        CV: int = 5,
        top_k: int = 1,
        estimators: List[str] = default_risk_estimation_names,
        feature_scaling: List[str] = default_feature_scaling_names,
        imputers: List[str] = [],
        hooks: Hooks = DefaultHooks(),
    ) -> None:
        self.time_horizons = time_horizons

        self.num_iter = num_iter
        self.timeout = timeout
        self.top_k = top_k
        self.study_name = study_name
        self.hooks = hooks

        self.CV = CV

        self.estimators = [
            PipelineSelector(
                estimator,
                classifier_category="risk_estimation",
                calibration=[],
                feature_selection=[],
                feature_scaling=feature_scaling,
                imputers=imputers,
            )
            for estimator in estimators
        ]

    def _should_continue(self) -> None:
        if self.hooks.cancel():
            raise StudyCancelled("risk estimation search cancelled")

    def search_best_args_for_estimator(
        self,
        estimator: Any,
        X: pd.DataFrame,
        T: pd.DataFrame,
        Y: pd.DataFrame,
        time_horizon: int,
    ) -> Tuple[float, float, Dict]:
        self._should_continue()

        def evaluate_estimator(**kwargs: Any) -> float:
            start = time.time()
            time_horizons = [time_horizon]

            model = estimator.get_pipeline_from_named_args(**kwargs)

            try:
                metrics = evaluate_survival_estimator(model, X, T, Y, time_horizons)
            except BaseException as e:
                log.error("evaluate_survival_estimator failed ", e)
                return 0

            self.hooks.heartbeat(
                topic="risk_estimation",
                subtopic="model_search",
                event_type="performance",
                name=model.name(),
                model_args=kwargs,
                duration=time.time() - start,
                horizon=time_horizon,
                aucroc=metrics["str"]["aucroc"],
                cindex=metrics["str"]["c_index"],
                brier_score=metrics["str"]["brier_score"],
            )
            return metrics["clf"]["c_index"][0] - metrics["clf"]["brier_score"][0]

        baseline_score = evaluate_estimator()

        if len(estimator.hyperparameter_space()) == 0:
            return baseline_score, baseline_score, {}

        log.info(f"baseline score for {estimator.name()} {baseline_score}")

        study, pruner = create_study(
            study_name=f"{self.study_name}_risk_estimation_exploration_{estimator.name()}_{time_horizon}",
        )

        def objective(trial: optuna.Trial) -> float:
            self._should_continue()
            pruner.check_patience(trial)

            args = estimator.sample_hyperparameters(trial)

            pruner.check_trial(trial)

            score = evaluate_estimator(**args)

            pruner.report_score(score)

            return score

        try:
            study.optimize(objective, n_trials=self.num_iter, timeout=self.timeout)
        except EarlyStoppingExceeded:
            log.info("Early stopping triggered for search")

        log.info(
            f"Best trial for estimator {estimator.name()}:{time_horizon}: {study.best_value} for {study.best_trial.params}"
        )

        return baseline_score, study.best_value, study.best_trial.params

    def search_estimator(
        self, X: pd.DataFrame, T: pd.DataFrame, Y: pd.DataFrame, time_horizon: int
    ) -> List:
        self._should_continue()

        log.info(f"Searching estimators for horizon {time_horizon}")
        try:
            search_results = dispatcher(
                delayed(self.search_best_args_for_estimator)(
                    estimator, X, T, Y, time_horizon
                )
                for estimator in self.estimators
            )
        except BaseException as e:
            print(traceback.format_exc())
            raise e

        all_scores = []
        all_args = []

        for idx, (baseline_score, best_score, best_args) in enumerate(search_results):
            all_scores.append([baseline_score, best_score])
            all_args.append([{}, best_args])

            log.info(
                f"Time horizon {time_horizon}: evaluation for {self.estimators[idx].name()} scores: baseline {baseline_score} optimized {best_score}. Args {best_args}"
            )

        all_scores_np = np.array(all_scores)
        selected_points = min(self.top_k, len(all_scores))
        best_scores = np.sort(np.unique(all_scores_np.ravel()))[-selected_points:]

        result = []
        for score in reversed(best_scores):
            pos = np.argwhere(all_scores_np == score)[0]
            pos_est = pos[0]
            est_args = pos[1]
            log.info(
                f"Selected score {score}: {self.estimators[pos_est].name()} : {all_args[pos_est][est_args]}"
            )
            result.append((pos_est, all_args[pos_est][est_args]))

        return result

    def search(self, X: pd.DataFrame, T: pd.DataFrame, Y: pd.DataFrame) -> List:
        self._should_continue()

        result = []
        for time_horizon in self.time_horizons:
            best_estimators_template = self.search_estimator(X, T, Y, time_horizon)
            horizon_result = []
            for idx, args in best_estimators_template:
                horizon_result.append(
                    self.estimators[idx].get_pipeline_from_named_args(**args)
                )
            result.append(horizon_result)

        return result
