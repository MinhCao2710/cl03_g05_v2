# MLOps Pipeline — Automated Retraining and Monitoring

Complete MLOps pipeline with automated retraining on new data push, drift monitoring, and centralised artifact storage via DVC and DagsHub.

---

## Project Structure

```
your-project/
├── .github/workflows/
│   └── train.yaml              # Auto-retraining workflow
├── src/
│   ├── model.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   ├── monitor.py              # Drift & performance monitoring
│   ├── preprocess_new_data.py  # Preprocess new data
│   └── retrain.py              # Incremental retraining script
├── data/
│   └── new_data.csv            # Upload new data here to trigger retraining
├── train/
│   └── train.csv               # Initial training data
├── test/
│   └── test.csv                # Test data
├── artifacts/                  # Centralised DVC-tracked storage
│   ├── data/
│   ├── preprocessing/
│   ├── metrics/
│   └── metadata/
├── models/
│   ├── base_model.keras        # Model trained from the original training set
│   └── model.keras             # Deployed model after the retraining decision
├── reports/
├── dvc.yaml                    # DVC pipeline definition
├── dvc.lock                    # DVC pipeline lock file (auto-updated by CI)
└── params.yaml                 # All configurable parameters
```

---

## Initial Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Configure DVC remote (DagsHub)

Run these commands locally. The `--local` flag keeps your credentials out of version control.

```bash
dvc remote add -d origin https://dagshub.com/YOUR_DAGSHUB_USERNAME/YOUR_REPO_NAME.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user YOUR_DAGSHUB_USERNAME
dvc remote modify origin --local password YOUR_DAGSHUB_TOKEN
```

> Your DagsHub token can be found at: https://dagshub.com/user/settings/tokens

### 4. Pull existing data and model artifacts

```bash
dvc pull
```

---

## Running the Pipeline

### Run full pipeline locally

```bash
dvc repro
```

DVC will only re-run stages whose dependencies have changed.

### Push artifacts to DagsHub after running

```bash
dvc push
```

### Pull latest artifacts from DagsHub

```bash
dvc pull
```

---

## Adding New Data to Trigger Retraining

1. Place your new data file in the `data/` folder:

```bash
cp /path/to/your/new_data.csv data/new_data.csv
```

2. Commit and push:

```bash
git add data/new_data.csv
git commit -m "feat(data): add new data batch for retraining"
git push
```

GitHub Actions will automatically detect the push and run the full pipeline.

---

## GitHub Actions Workflow

The workflow (`.github/workflows/train.yaml`) runs on three triggers:

| Trigger | When |
|---|---|
| `push` | Pushes to `main` that change new data, pipeline code, or configuration |
| `schedule` | Pipeline check every Sunday at midnight (UTC) |
| `workflow_dispatch` | Manually from the GitHub Actions tab |

### Required GitHub Secrets

Go to **Settings → Secrets and variables → Actions → New repository secret** and add:

| Secret Name | Description |
|---|---|
| `DAGSHUB_USERNAME` | Your DagsHub username |
| `DAGSHUB_TOKEN` | Your DagsHub access token (from Account Settings → Tokens) |

> **Never commit your token or password.** Always use the `--local` flag for DVC credentials locally, and GitHub secrets in CI.

---

## Pipeline Stages

Defined in `dvc.yaml`:

1. **train** — trains the base Keras classifier on the original training set and saves preprocessing artifacts
2. **preprocess_new_data** — validates, cleans, and merges incoming data with the training set; outputs numpy arrays to `artifacts/data/`
3. **monitor** — checks data quality, drift, and performance; writes metrics and `artifacts/metadata/retraining_flags.json`
4. **retrain** — fine-tunes the base model only when monitoring requests retraining; otherwise promotes it unchanged
5. **evaluate** — computes accuracy, precision, recall, and F1 for the deployed model

---

## Configuration

All tunable parameters live in `params.yaml`. Edit this file to change model hyperparameters, drift thresholds, or preprocessing options without touching source code.

```yaml
training:
  epochs: 120
  batch_size: 32

monitoring:
  accuracy_drop_threshold: 0.05
  missing_rate_threshold: 0.05
```

---

## Team

| Member | Responsibility |
|---|---|
| Member 1 | DVC setup, GitHub Actions CI/CD, remote storage |
| Member 2 | Monitoring, data quality checks, drift detection |
| Member 3 | Model training, evaluation, retraining logic |
