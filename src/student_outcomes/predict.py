"""Predict student outcomes from a saved estimator and local enrollment CSV."""

import argparse
import json
from pathlib import Path

from joblib import load

from student_outcomes.data import CLASS_ORDER, load_prediction_features


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict enrollment-time outcomes.")
    parser.add_argument("--model", type=Path, required=True, help="Saved joblib model")
    parser.add_argument("--data", type=Path, required=True, help="Enrollment CSV")
    args = parser.parse_args(argv)

    try:
        features = load_prediction_features(args.data)
        model = load(args.model)
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    labels = [CLASS_ORDER[code] for code in model.classes_]
    print(
        json.dumps(
            [
                {
                    "prediction": CLASS_ORDER[prediction],
                    "probabilities": dict(zip(labels, map(float, scores), strict=True)),
                }
                for prediction, scores in zip(predictions, probabilities, strict=True)
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
