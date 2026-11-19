# Predict Students’ Dropout and Academic Success

[![CI](https://github.com/vivalerievich/predict-students-dropout-and-academic-success/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/vivalerievich/predict-students-dropout-and-academic-success/actions/workflows/ci.yml)

**Task.** Predict `Dropout`, `Enrolled`, or `Graduate` at the end of enrollment,
before first-semester teaching and assessment.

**Result.** Logistic regression achieved **0.4918 macro F1** on the 885-student
holdout and identified **141 of 284** students with the `Dropout` outcome.

**Main limitation.** The evaluation covers a random holdout from one
institution. Performance on later cohorts and at other institutions remains
unmeasured.

The dataset contains 4,424 student records and 36 original predictors.
[UCI dataset](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

## Run predictions

Clone the code at the model's release tag and install its locked environment.

```bash
git clone --branch v0.1.0 https://github.com/vivalerievich/predict-students-dropout-and-academic-success.git
cd predict-students-dropout-and-academic-success
uv sync --locked --no-dev --python 3.14
mkdir -p artifacts
```

Download `predict-students-dropout-and-academic-success.joblib`,
`sample-input.csv` and `SHA256SUMS.txt` into `artifacts/`. The fitted model
includes preprocessing and the logistic regression estimator. Use joblib
models from sources you trust. [GitHub Releases](https://github.com/vivalerievich/predict-students-dropout-and-academic-success/releases/latest)

```bash
(cd artifacts && shasum -a 256 -c SHA256SUMS.txt)
uv run --locked --no-dev student-outcomes-predict \
  --model artifacts/predict-students-dropout-and-academic-success.joblib \
  --data artifacts/sample-input.csv
```

Use the synthetic sample as the input schema. The semicolon-separated CSV
requires the 18 enrollment fields. The CLI emits a JSON array in input row
order with a predicted label and a probability for each of the three classes.

## Run the benchmark

Install uv and run the commands from the repository root. The result table
below uses Python 3.14. [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv sync
mkdir -p data
curl -fL 'https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip' -o data/uci-697.zip
unzip -o data/uci-697.zip -d data
uv run student-outcomes-train --data data/data.csv
```

The command prints a JSON report with validation scores, holdout metrics by
class and confusion matrices. The CSV inside the UCI archive uses semicolons
as delimiters. Keep the archive and CSV under the ignored `data/` directory.

Add `--model-out artifacts/model.joblib` to export the fitted estimator
selected by mean cross-validation macro F1. Its fit uses the training split.

The default benchmark compares a majority-class baseline, logistic
regression, a decision tree, random forest and gradient boosting. To reproduce
the seven-model comparison in the notebook, add XGBoost and LightGBM.
On macOS, install OpenMP first with `brew install libomp`.
[LightGBM installation guide](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html)

```bash
uv run --extra extended student-outcomes-train --data data/data.csv --extended
```

For exploratory analysis, search plots and feature importances, launch
JupyterLab and run
[`notebooks/predict-students-dropout-and-academic-success.ipynb`](notebooks/predict-students-dropout-and-academic-success.ipynb).

```bash
uv run --extra extended --group notebook jupyter lab
```

## Evaluation

The stratified 80/20 split uses seed 42, with 3,539 students for development and
885 for holdout evaluation. Model selection uses mean macro F1 across five
stratified folds of the development set. Every fold fits its own imputation
and encoding. Logistic regression also scales the numerical variables. The
decision tree and logistic regression use grid search. Each ensemble uses
24 seeded Optuna trials.

| Model | CV macro F1 (mean ± SD) | Holdout macro F1 |
| --- | ---: | ---: |
| **Logistic regression** | **0.5191 ± 0.0202** | **0.4918** |
| Random forest | 0.5172 ± 0.0177 | 0.4857 |
| XGBoost (extended) | 0.5116 ± 0.0129 | 0.4721 |
| Gradient boosting | 0.5060 ± 0.0159 | 0.4640 |
| LightGBM (extended) | 0.5031 ± 0.0208 | 0.4763 |
| Decision tree | 0.4703 ± 0.0139 | 0.4544 |
| Majority baseline | 0.2220 ± 0.0002 | 0.2221 |

Logistic regression had the highest mean CV macro F1. On the holdout, its
weighted F1 was 0.5389 and balanced accuracy was 0.4992. It identified 141
of the 284 students with the `Dropout` outcome, with precision 0.5402 and
recall 0.4965.

## Limitations

The training pipeline explicitly selects 18 enrollment-time predictors. It
excludes semester grades, evaluations, enrollments, credits and other
semester summaries. It also excludes debt, tuition-payment, scholarship and
macroeconomic fields because their availability at the end of enrollment is
uncertain in the released data.

`Enrolled` records the observed status at the study endpoint. The results
describe one institution on a random holdout. Performance on later cohorts,
at other institutions and across demographic groups remains unmeasured.

## Checks

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

The tests generate small local fixtures and run without downloading the dataset.

The [model release smoke test](https://github.com/vivalerievich/predict-students-dropout-and-academic-success/actions/workflows/model-smoke.yml)
runs on release publication and can be run manually for a published tag. It
loads the released model with the tagged code and checks the synthetic example
against the contract in `tests/release/<tag>/`. Add that tag's
`schema.json` and `expected-output.json` before publishing a new model release.

Data by Valentim Realinho, Mónica Vieira Martins, Jorge Machado and Luís
Baptista (2021). [UCI DOI 10.24432/C5MC89](https://doi.org/10.24432/C5MC89).
The dataset is available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
