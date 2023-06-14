# adjutorium absolute
from adjutorium.plugins.preprocessors import Preprocessors

default_classifiers_names = [
    "random_forest",
    "xgboost",
    "adaboost",
    "bagging",
    "qda",
    "lda",
    "logistic_regression",
    "lgbm",
    "catboost",
]
default_imputers_names = [
    "nop",
    "median",
    "mean",
    "most_frequent",
]
default_feature_scaling_names = Preprocessors(category="feature_scaling").list()
default_feature_selection_names = ["nop", "variance_threshold", "pca", "fast_ica"]
default_risk_estimation_names = [
    "survival_catboost",
    "survival_xgboost",
    "cox_ph_ridge",
    "loglogistic_aft",
    "deephit",
    "cox_ph",
    "weibull_aft",
    "lognormal_aft",
    "coxnet",
]

percentile_val = 1.96
