"""Input schema and enrollment-time prediction boundary."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Target"
CLASS_ORDER = ("Dropout", "Enrolled", "Graduate")
CLASS_CODES = {label: index for index, label in enumerate(CLASS_ORDER)}
RANDOM_STATE = 42
TEST_SIZE = 0.20

ENROLLMENT_FEATURES = (
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Previous qualification (grade)",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Gender",
    "Age at enrollment",
    "International",
)
NUMERIC_FEATURES = (
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
)
CATEGORICAL_FEATURES = tuple(
    column for column in ENROLLMENT_FEATURES if column not in NUMERIC_FEATURES
)


def load_dataset(path: Path) -> pd.DataFrame:
    """Read the official UCI archive's semicolon-separated CSV."""
    data = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    data.columns = data.columns.str.strip()
    missing = sorted(set((*ENROLLMENT_FEATURES, TARGET)) - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if data[TARGET].isna().any() or set(data[TARGET]) != set(CLASS_ORDER):
        raise ValueError(f"{TARGET} must contain all three UCI outcome labels")
    return data


def split_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return disjoint, stratified train and holdout frames."""
    return train_test_split(
        data, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=data[TARGET]
    )


def extract_inputs(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Keep only features available at enrollment and encode the outcomes."""
    return data.loc[:, list(ENROLLMENT_FEATURES)], data[TARGET].map(CLASS_CODES)
