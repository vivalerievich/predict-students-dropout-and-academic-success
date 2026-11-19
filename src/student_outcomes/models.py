"""Preprocessing and fold-local model selection."""

from dataclasses import dataclass
from tempfile import TemporaryDirectory

import numpy as np
import optuna
import pandas as pd
from joblib import Memory, parallel_config
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from student_outcomes.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE

BASELINE_NAME = "Majority baseline"
SCORING = "f1_macro"
CV_FOLDS = 5
DEFAULT_TRIALS = 24


@dataclass
class FittedModel:
    estimator: BaseEstimator
    fold_scores: np.ndarray
    params: dict
    search: GridSearchCV | optuna.Study | None = None

    @property
    def cv_mean(self) -> float:
        return float(self.fold_scores.mean())

    @property
    def cv_std(self) -> float:
        return float(self.fold_scores.std())


def make_pipeline(classifier, *, scale_numeric=False, memory=None) -> Pipeline:
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ]
    )
    return Pipeline(
        [("preprocessor", preprocessor), ("classifier", classifier)], memory=memory
    )


def suggest_forest_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 24),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 24, log=True),
        "max_features": trial.suggest_float("max_features", 0.25, 1.0),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def suggest_gradient_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 60, 220, step=20),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.20, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 4),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 30, log=True),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
    }


def suggest_xgb_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.20, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 7),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 16, log=True),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20, log=True),
    }


def suggest_lgbm_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.20, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 31, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 80, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20, log=True),
    }


def _ensemble_specs(extended):
    specs = {
        "Random forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            suggest_forest_params,
        ),
        "Gradient boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            suggest_gradient_params,
        ),
    }
    if extended:
        try:
            from lightgbm import LGBMClassifier
            from xgboost import XGBClassifier
        except ModuleNotFoundError as error:
            raise ValueError(
                "Install the extended dependencies with uv sync --extra extended"
            ) from error
        specs.update(
            {
                "XGBoost": (
                    XGBClassifier(
                        random_state=RANDOM_STATE,
                        tree_method="hist",
                        n_jobs=1,
                        importance_type="total_gain",
                    ),
                    suggest_xgb_params,
                ),
                "LightGBM": (
                    LGBMClassifier(
                        random_state=RANDOM_STATE,
                        verbosity=-1,
                        n_jobs=1,
                        importance_type="gain",
                    ),
                    suggest_lgbm_params,
                ),
            }
        )
    return specs


def _baseline_grids(memory):
    return {
        "Logistic regression": (
            make_pipeline(
                LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
                scale_numeric=True,
                memory=memory,
            ),
            {
                "classifier__C": [0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "Decision tree": (
            make_pipeline(
                DecisionTreeClassifier(random_state=RANDOM_STATE), memory=memory
            ),
            {
                "classifier__max_depth": [1, 3, 5, 8, 20, None],
                "classifier__min_samples_leaf": [1, 5, 10, 25, 50],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
    }


def _fit_grid(pipeline, params, features, targets, folds, jobs):
    search = GridSearchCV(
        pipeline, params, scoring=SCORING, cv=folds, n_jobs=jobs, error_score="raise"
    ).fit(features, targets)
    scores = np.array(
        [
            search.cv_results_[f"split{fold}_test_score"][search.best_index_]
            for fold in range(len(folds))
        ]
    )
    best_params = {
        key.removeprefix("classifier__"): value
        for key, value in search.best_params_.items()
    }
    search.best_estimator_.memory = None
    return FittedModel(search.best_estimator_, scores, best_params, search)


def _fit_ensemble(spec, memory, features, targets, folds, jobs, trials):
    classifier, suggest_params = spec

    def build_model(params):
        return make_pipeline(clone(classifier).set_params(**params), memory=memory)

    def objective(trial):
        scores = cross_val_score(
            build_model(suggest_params(trial)),
            features,
            targets,
            scoring=SCORING,
            cv=folds,
            n_jobs=jobs,
            error_score="raise",
        )
        trial.set_user_attr("fold_scores", scores.tolist())
        return float(scores.mean())

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=6)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=trials)
    model = build_model(study.best_params).fit(features, targets)
    model.memory = None
    scores = np.asarray(study.best_trial.user_attrs["fold_scores"])
    return FittedModel(model, scores, study.best_params, study)


def fit_models(
    features: pd.DataFrame,
    targets: pd.Series,
    *,
    trials: int = DEFAULT_TRIALS,
    jobs: int = 1,
    extended: bool = False,
) -> dict[str, FittedModel]:
    """Fit candidates on training rows and rank them by cross-validation macro F1."""
    ensemble_specs = _ensemble_specs(extended)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(features, targets))
    with TemporaryDirectory(prefix="student-cv-") as cache:
        memory = Memory(cache, verbose=0)
        with parallel_config(backend="threading"), threadpool_limits(limits=1):
            dummy = DummyClassifier(strategy="most_frequent")
            scores = cross_val_score(
                dummy,
                features,
                targets,
                scoring=SCORING,
                cv=folds,
                n_jobs=jobs,
                error_score="raise",
            )
            models = {
                BASELINE_NAME: FittedModel(
                    dummy.fit(features, targets), scores, {"strategy": "most_frequent"}
                )
            }
            for name, (pipeline, params) in _baseline_grids(memory).items():
                models[name] = _fit_grid(
                    pipeline, params, features, targets, folds, jobs
                )
            for name, spec in ensemble_specs.items():
                models[name] = _fit_ensemble(
                    spec, memory, features, targets, folds, jobs, trials
                )
    return dict(
        sorted(models.items(), key=lambda item: (-item[1].cv_mean, item[1].cv_std))
    )
