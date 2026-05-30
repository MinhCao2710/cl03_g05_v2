"""Prepare incoming data for classifier inference/monitoring and retraining."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_params() -> dict[str, Any]:
    with project_path("params.yaml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def label_key(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        float_value = float(value)
        return str(int(float_value)) if float_value.is_integer() else str(float_value)
    return str(value)


def align_features(dataset: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    missing_columns = [column for column in feature_columns if column not in dataset.columns]
    extra_columns = [column for column in dataset.columns if column not in feature_columns]
    aligned = dataset.reindex(columns=feature_columns).copy()

    for column in feature_columns:
        aligned[column] = pd.to_numeric(aligned[column], errors="coerce")

    return aligned, missing_columns, extra_columns


def encode_target_series(series: pd.Series, class_labels: list[Any]) -> np.ndarray:
    label_to_index = {label_key(label): index for index, label in enumerate(class_labels)}
    encoded = np.full((len(series),), np.nan, dtype=np.float32)

    for row_index, value in enumerate(series):
        key = label_key(value)
        if key in label_to_index:
            encoded[row_index] = label_to_index[key]

    return encoded


def save_combined_training_data(
    train_df: pd.DataFrame,
    new_df: pd.DataFrame,
    target_column: str,
    class_labels: list[Any],
    combined_path: Path,
) -> int:
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    if target_column not in new_df.columns:
        train_df.to_csv(combined_path, index=False)
        return 0

    new_for_training = new_df.reindex(columns=train_df.columns).copy()
    valid_label_keys = {label_key(label) for label in class_labels}
    labelled_mask = new_for_training[target_column].map(label_key).isin(valid_label_keys)
    labelled_new_rows = new_for_training.loc[labelled_mask]
    combined_df = pd.concat([train_df, labelled_new_rows], ignore_index=True)
    combined_df.to_csv(combined_path, index=False)
    return int(len(labelled_new_rows))


def main() -> None:
    params = load_params()
    paths = params["paths"]

    feature_info = load_json(project_path(paths["feature_info_path"]))
    feature_columns = feature_info["feature_columns"]
    target_column = feature_info["target_column"]
    class_labels = feature_info["class_labels"]
    preprocessor = joblib.load(project_path(paths["scaler_path"]))

    new_data_path = project_path(paths["new_data"])
    if not new_data_path.exists():
        raise FileNotFoundError(f"New data file not found: {new_data_path}")

    new_df = pd.read_csv(new_data_path)
    X_new_raw, missing_columns, extra_columns = align_features(new_df, feature_columns)
    extra_columns = [column for column in extra_columns if column != target_column]
    X_new_scaled = preprocessor.transform(X_new_raw).astype(np.float32)
    X_new_cnn = X_new_scaled.reshape((X_new_scaled.shape[0], X_new_scaled.shape[1], 1))

    y_new = np.full((len(new_df),), np.nan, dtype=np.float32)
    labelled_rows = 0
    if target_column in new_df.columns:
        y_new = encode_target_series(new_df[target_column], class_labels)
        labelled_rows = int(np.isfinite(y_new).sum())

    X_new_path = project_path(paths["x_new_path"])
    X_new_2d_path = project_path(paths["x_new_2d_path"])
    y_new_path = project_path(paths["y_new_path"])
    aligned_new_path = project_path(paths["aligned_new_data_path"])
    for path in [X_new_path, X_new_2d_path, y_new_path, aligned_new_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    np.save(X_new_path, X_new_cnn)
    np.save(X_new_2d_path, X_new_scaled)
    np.save(y_new_path, y_new)
    aligned_payload = X_new_raw.copy()
    if target_column in new_df.columns:
        aligned_payload[target_column] = new_df[target_column].values
    aligned_payload.to_csv(aligned_new_path, index=False)

    train_df = pd.read_csv(project_path(paths["train_data"]))
    appended_rows = save_combined_training_data(
        train_df,
        new_df,
        target_column,
        class_labels,
        project_path(paths["combined_train_data"]),
    )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "new_data_path": paths["new_data"],
        "rows": int(len(new_df)),
        "labelled_rows": labelled_rows,
        "appended_rows_for_retraining": appended_rows,
        "missing_feature_columns": missing_columns,
        "extra_input_columns": extra_columns,
        "class_labels": class_labels,
        "x_new_shape": list(X_new_cnn.shape),
        "x_new_2d_shape": list(X_new_scaled.shape),
    }
    save_json(project_path(paths["new_data_metadata_path"]), metadata)

    print("New classification data preprocessing complete")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
