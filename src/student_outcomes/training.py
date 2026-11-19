"""Run and evaluate the enrollment-time benchmark."""

from pathlib import Path

from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from student_outcomes.data import (
    CLASS_CODES,
    CLASS_ORDER,
    extract_inputs,
    load_dataset,
    split_dataset,
)
from student_outcomes.models import DEFAULT_TRIALS, fit_models


def run_benchmark(
    path: Path, *, trials: int = DEFAULT_TRIALS, jobs: int = 1, extended: bool = False
) -> dict:
    train, holdout = split_dataset(load_dataset(path))
    train_features, train_targets = extract_inputs(train)
    holdout_features, holdout_targets = extract_inputs(holdout)
    models = fit_models(
        train_features, train_targets, trials=trials, jobs=jobs, extended=extended
    )
    class_codes = list(CLASS_CODES.values())
    results = {}
    for name, model in models.items():
        predictions = model.estimator.predict(holdout_features)
        report = classification_report(
            holdout_targets,
            predictions,
            labels=class_codes,
            target_names=CLASS_ORDER,
            output_dict=True,
            zero_division=0,
        )
        results[name] = {
            "parameters": model.params,
            "cv_macro_f1_mean": model.cv_mean,
            "cv_macro_f1_std": model.cv_std,
            "holdout_macro_f1": report["macro avg"]["f1-score"],
            "holdout_weighted_f1": report["weighted avg"]["f1-score"],
            "holdout_balanced_accuracy": float(
                balanced_accuracy_score(holdout_targets, predictions)
            ),
            "holdout_precision_recall_by_class": {
                label: report[label] for label in CLASS_ORDER
            },
            "holdout_confusion_matrix": confusion_matrix(
                holdout_targets, predictions, labels=class_codes
            ).tolist(),
        }
    return {
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "class_order": CLASS_ORDER,
        "selected_model": next(iter(models)),
        "results": results,
    }
