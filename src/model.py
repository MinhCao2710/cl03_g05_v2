"""Train a TensorFlow MLP multi-class classifier and save reproducible artifacts."""

from __future__ import annotations

import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_params() -> dict[str, Any]:
    with project_path("params.yaml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=json_default)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def read_dataset(path: Path, target_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataset = pd.read_csv(path)
    if target_column not in dataset.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found in {path}. "
            f"Available columns: {list(dataset.columns)}"
        )
    return dataset


def to_python_label(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        float_value = float(value)
        return int(float_value) if float_value.is_integer() else float_value
    return value


def split_features_target(
    dataset: pd.DataFrame, target_column: str, feature_columns: list[str] | None = None
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    if feature_columns is None:
        feature_columns = [column for column in dataset.columns if column != target_column]

    X = dataset.reindex(columns=feature_columns).copy()
    for column in feature_columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    y_numeric = pd.to_numeric(dataset[target_column], errors="coerce")
    if y_numeric.isna().any():
        missing_count = int(y_numeric.isna().sum())
        raise ValueError(f"Target column '{target_column}' contains {missing_count} missing values.")

    if np.allclose(y_numeric, np.round(y_numeric)):
        y = y_numeric.astype(int).to_numpy()
    else:
        y = y_numeric.to_numpy()

    y = np.array([to_python_label(value) for value in y], dtype=object)
    return X, y, feature_columns


def make_class_mapping(*label_arrays: np.ndarray) -> tuple[list[Any], dict[Any, int]]:
    labels = pd.unique(np.concatenate(label_arrays)).tolist()
    labels = [to_python_label(label) for label in labels]
    labels = sorted(labels, key=lambda label: (str(type(label)), label))
    class_to_index = {label: index for index, label in enumerate(labels)}
    return labels, class_to_index


def encode_labels(labels: np.ndarray, class_to_index: dict[Any, int]) -> np.ndarray:
    return np.array([class_to_index[to_python_label(label)] for label in labels], dtype=np.int64)


def build_preprocessor(feature_columns: list[str], params: dict[str, Any]) -> ColumnTransformer:
    preprocessing_params = params.get("preprocessing", {})
    imputer_strategy = preprocessing_params.get("imputer_strategy", "median")

    standard_columns = [
        column
        for column in preprocessing_params.get("standard_scale_columns", [])
        if column in feature_columns
    ]
    minmax_columns = [
        column
        for column in preprocessing_params.get("minmax_scale_columns", [])
        if column in feature_columns
    ]
    scaled_columns = set(standard_columns + minmax_columns)
    passthrough_columns = [column for column in feature_columns if column not in scaled_columns]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if standard_columns:
        transformers.append(
            (
                "standard",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy=imputer_strategy)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                standard_columns,
            )
        )

    if minmax_columns:
        transformers.append(
            (
                "minmax",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy=imputer_strategy)),
                        ("scaler", MinMaxScaler()),
                    ]
                ),
                minmax_columns,
            )
        )

    if passthrough_columns:
        transformers.append(
            (
                "passthrough_numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy=imputer_strategy))]),
                passthrough_columns,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        names: list[str] = []
        for transformer_name, _, columns in preprocessor.transformers_:
            if transformer_name == "remainder":
                continue
            names.extend([f"{transformer_name}__{column}" for column in columns])
        return names


def build_model(input_dim: int, num_classes: int, params: dict[str, Any]) -> keras.Model:
    model_params = params["model"]
    regularizer = None
    l2_reg = float(model_params.get("l2_reg", 0.0))
    if l2_reg > 0:
        regularizer = keras.regularizers.l2(l2_reg)

    model = keras.Sequential(name=model_params.get("name", "mlp_classifier"))
    model.add(keras.layers.Input(shape=(input_dim,), name="features"))

    for index, units in enumerate(model_params.get("hidden_layers", [64, 32]), start=1):
        model.add(
            keras.layers.Dense(
                int(units),
                activation=model_params.get("activation", "relu"),
                kernel_regularizer=regularizer,
                name=f"dense_{index}",
            )
        )
        dropout_rate = float(model_params.get("dropout", 0.0))
        if dropout_rate > 0:
            model.add(keras.layers.Dropout(dropout_rate, name=f"dropout_{index}"))

    model.add(keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities"))

    optimizer = keras.optimizers.Adam(learning_rate=float(model_params.get("learning_rate", 0.001)))
    model.compile(
        optimizer=optimizer,
        loss=model_params.get("loss", "sparse_categorical_crossentropy"),
        metrics=["accuracy"],
    )
    return model


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def main() -> None:
    params = load_params()
    paths = params["paths"]
    target_column = params["data"]["target_column"]
    seed = int(params["training"].get("random_state", 42))
    set_random_seed(seed)

    train_path = project_path(paths["train_data"])
    test_path = project_path(paths["test_data"])
    train_df = read_dataset(train_path, target_column)
    test_df = read_dataset(test_path, target_column)

    X_train_raw, y_train_labels, feature_columns = split_features_target(train_df, target_column)
    X_test_raw, y_test_labels, _ = split_features_target(test_df, target_column, feature_columns)
    class_labels, class_to_index = make_class_mapping(y_train_labels, y_test_labels)
    y_train = encode_labels(y_train_labels, class_to_index)
    y_test = encode_labels(y_test_labels, class_to_index)

    preprocessor = build_preprocessor(feature_columns, params)
    X_train = preprocessor.fit_transform(X_train_raw).astype(np.float32)
    X_test = preprocessor.transform(X_test_raw).astype(np.float32)

    model = build_model(input_dim=X_train.shape[1], num_classes=len(class_labels), params=params)
    training_params = params["training"]
    callbacks: list[keras.callbacks.Callback] = []
    early_stopping_params = training_params.get("early_stopping", {})
    if early_stopping_params.get("enabled", True):
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=early_stopping_params.get("monitor", "val_loss"),
                patience=int(early_stopping_params.get("patience", 20)),
                restore_best_weights=bool(early_stopping_params.get("restore_best_weights", True)),
            )
        )

    history = model.fit(
        X_train,
        y_train,
        epochs=int(training_params.get("epochs", 100)),
        batch_size=int(training_params.get("batch_size", 32)),
        validation_split=float(training_params.get("validation_split", 0.2)),
        callbacks=callbacks,
        verbose=int(training_params.get("verbose", 1)),
    )

    train_probabilities = model.predict(X_train, verbose=0)
    test_probabilities = model.predict(X_test, verbose=0)
    y_train_pred = np.argmax(train_probabilities, axis=1)
    y_test_pred = np.argmax(test_probabilities, axis=1)
    train_metrics = calculate_classification_metrics(y_train, y_train_pred)
    test_metrics = calculate_classification_metrics(y_test, y_test_pred)

    model_path = project_path(paths.get("base_model_path", paths["model_path"]))
    legacy_model_path = project_path(paths.get("base_legacy_model_path", paths["legacy_model_path"]))
    scaler_path = project_path(paths["scaler_path"])
    feature_columns_path = project_path(paths["feature_columns_path"])
    feature_info_path = project_path(paths["feature_info_path"])
    history_path = project_path(paths["training_history_path"])
    train_metrics_path = project_path(paths["train_metrics_path"])
    test_metrics_path = project_path(paths["test_metrics_path"])
    metadata_path = project_path(paths["training_metadata_path"])
    x_test_path = project_path(paths["x_test_path"])
    y_test_path = project_path(paths["y_test_path"])

    model_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    model.save(legacy_model_path)

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, scaler_path)

    feature_columns_path.parent.mkdir(parents=True, exist_ok=True)
    with feature_columns_path.open("w", encoding="utf-8") as file:
        json.dump(feature_columns, file, indent=2)

    feature_info = {
        "target_column": target_column,
        "feature_columns": feature_columns,
        "transformed_feature_columns": transformed_feature_names(preprocessor),
        "class_labels": class_labels,
        "class_to_index": {str(label): int(index) for label, index in class_to_index.items()},
        "standard_scale_columns": params["preprocessing"].get("standard_scale_columns", []),
        "minmax_scale_columns": params["preprocessing"].get("minmax_scale_columns", []),
    }
    save_json(feature_info_path, feature_info)

    history_payload = {
        metric: [float(value) for value in values] for metric, values in history.history.items()
    }
    save_json(history_path, history_payload)
    save_json(train_metrics_path, {"rows": int(len(y_train)), "classes": len(class_labels), **train_metrics})
    save_json(test_metrics_path, {"rows": int(len(y_test)), "classes": len(class_labels), **test_metrics})

    x_test_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(x_test_path, X_test)
    np.save(y_test_path, y_test)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "pandas_version": pd.__version__,
        "random_state": seed,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "input_dim": int(X_train.shape[1]),
        "num_classes": int(len(class_labels)),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "legacy_model_path": str(legacy_model_path.relative_to(PROJECT_ROOT)),
        "scaler_path": str(scaler_path.relative_to(PROJECT_ROOT)),
    }
    save_json(metadata_path, metadata)

    print("Classification training complete")
    print(json.dumps({"train_metrics": train_metrics, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
