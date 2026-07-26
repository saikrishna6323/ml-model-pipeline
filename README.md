# ml-model-pipeline

Training pipeline for a CI/CD pipeline-failure classifier. This is a
trainable, non-LLM alternative to the `LLMPredictor` in
[predictive-cicd-remediation](https://github.com/saikrishna6323/predictive-cicd-remediation),
useful when you want a lightweight model instead of calling an external LLM API.

## Contents

- `train.py` - trains a gradient-boosted classifier on historical pipeline-run features.
- `serve.py` - a small FastAPI service that loads the trained model and exposes `/predict`.
- `data/sample_runs.csv` - a documented synthetic dataset (clearly labeled as synthetic) used for demo training.

## Features used

`rolling_failure_rate`, `dependency_files_changed`, `step_duration_seconds`,
`num_recent_errors`, `day_of_week`, `hour_of_day` - the same feature shape
produced by `src/features/feature_extractor.py` in predictive-cicd-remediation,
so a trained model here can be dropped into that project's prediction module.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
python serve.py
```

## Status

Demo/research pipeline trained on a synthetic dataset. Replace
`data/sample_runs.csv` with real historical run data before using in
production.
