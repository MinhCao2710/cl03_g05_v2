# CL03_G05 MLOps Pipeline

This project turns the original notebook workflow into a reproducible MLOps-style pipeline for training, evaluating, preprocessing new data, and monitoring a TensorFlow MLP classification model.

## Folder Structure

```text
.
├── src/
│   ├── model.py
│   ├── evaluate.py
│   ├── monitor.py
│   └── preprocess_new_data.py
├── train/train.csv
├── test/test.csv
├── data/
│   ├── cleaned_dataset.csv
│   └── new_data.csv
├── artifacts/
│   ├── data/
│   ├── metrics/
│   ├── preprocessing/
│   └── metadata/
├── models/
├── logs/
├── reports/
├── params.yaml
├── dvc.yaml
├── requirements.txt
└── traditional_model.ipynb
```

## Setup

```bash
conda activate minhc_env
pip install -r requirements.txt
```

Use `tf_gpu_env` instead if you want to run TensorFlow with your GPU environment.

## Run Without DVC

```bash
python src/model.py
python src/evaluate.py
python src/preprocess_new_data.py
python src/monitor.py
```

## Run With DVC

```bash
dvc init
dvc repro
dvc metrics show
```

If DVC is already initialized in your repository, skip `dvc init`.

## Main Outputs

```text
models/model.keras
models/model.h5
artifacts/preprocessing/scaler.joblib
artifacts/preprocessing/feature_columns.json
artifacts/preprocessing/feature_info.json
artifacts/metrics/train_metrics.json
artifacts/metrics/test_metrics.json
artifacts/metrics/evaluation_metrics.json
artifacts/metrics/classification_report.json
artifacts/metrics/monitoring_metrics.json
artifacts/data/test_predictions.csv
artifacts/data/submission.csv
reports/confusion_matrix.png
reports/prediction_distribution.png
```

## What Stays In The Notebook

Keep exploratory work in `traditional_model.ipynb`: EDA, quick charts, target/class checks, feature experiments, and comparison notes for the assignment report.

Move repeatable work into Python scripts: data loading, preprocessing, classifier training, evaluation, artifact saving, new-data preparation, and monitoring decisions. This makes the workflow easier to rerun with DVC and easier to maintain.

## Retraining And Monitoring

`src/preprocess_new_data.py` aligns `data/new_data.csv` to the training feature columns, applies the saved scaler, saves NumPy arrays, and creates `artifacts/data/train_with_new_data.csv` when the new data includes labels.

`src/monitor.py` compares old test-set classification performance with labelled new-data performance. If accuracy, precision, recall, or F1 drops beyond the configured thresholds, it writes retraining flags to `artifacts/metadata/retraining_flags.json`.
