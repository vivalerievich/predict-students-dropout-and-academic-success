import json

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from student_outcomes.cli import main
from student_outcomes.data import (
    CLASS_ORDER,
    ENROLLMENT_FEATURES,
    TARGET,
    extract_inputs,
    load_dataset,
    split_dataset,
)
from student_outcomes.models import make_pipeline


@pytest.fixture
def dataset_path(tmp_path):
    rows = []
    for label_code, label in enumerate(CLASS_ORDER):
        for number in range(20):
            row = {feature: number % 3 for feature in ENROLLMENT_FEATURES}
            row["Admission grade"] = 100 + 20 * label_code + number
            row["Curricular units 1st sem (grade)"] = 100 + label_code
            row["Tuition fees up to date"] = label_code
            row[TARGET] = label
            rows.append(row)
    data = pd.DataFrame(rows).rename(
        columns={"Daytime/evening attendance": "Daytime/evening attendance\t"}
    )
    path = tmp_path / "data.csv"
    data.to_csv(path, sep=";", encoding="utf-8-sig", index=False)
    return path


def test_loads_official_csv_schema_and_selects_only_enrollment_features(dataset_path):
    data = load_dataset(dataset_path)
    features, targets = extract_inputs(data)

    assert list(features) == list(ENROLLMENT_FEATURES)
    assert targets.value_counts().to_dict() == {0: 20, 1: 20, 2: 20}


def test_missing_columns_raise_a_clear_error(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_text("Target\nDropout\n")

    with pytest.raises(ValueError, match="Missing columns:.*Admission grade"):
        load_dataset(path)


def test_split_is_disjoint_and_stratified(dataset_path):
    train, holdout = split_dataset(load_dataset(dataset_path))

    assert train.index.intersection(holdout.index).empty
    assert train[TARGET].value_counts().to_dict() == dict.fromkeys(CLASS_ORDER, 16)
    assert holdout[TARGET].value_counts().to_dict() == dict.fromkeys(CLASS_ORDER, 4)


def test_pipeline_handles_unknown_category_and_missing_grade(dataset_path):
    features, targets = extract_inputs(load_dataset(dataset_path))
    model = make_pipeline(LogisticRegression(max_iter=3000), scale_numeric=True)
    model.fit(features, targets)

    new_student = features.iloc[[0]].copy()
    new_student["Course"] = 9999
    new_student["Admission grade"] = float("nan")

    assert model.predict(new_student)[0] in {0, 1, 2}


def test_cli_trains_and_reports_metrics_from_local_csv(dataset_path, capsys):
    main(["--data", str(dataset_path), "--trials", "1", "--jobs", "1"])
    report = json.loads(capsys.readouterr().out)

    assert report["train_rows"] == 48
    assert report["holdout_rows"] == 12
    results = report["results"]
    assert report["selected_model"] == next(iter(results))
    assert results[report["selected_model"]]["cv_macro_f1_mean"] == max(
        result["cv_macro_f1_mean"] for result in results.values()
    )
    assert set(results) == {
        "Majority baseline",
        "Logistic regression",
        "Decision tree",
        "Random forest",
        "Gradient boosting",
    }
    assert all(
        sum(map(sum, result["holdout_confusion_matrix"])) == 12
        for result in results.values()
    )
