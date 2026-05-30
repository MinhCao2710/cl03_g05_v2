"""Retrain or fine-tune the classifier on the merged training dataset."""

from __future__ import annotations

import json
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from model import (
    PROJECT_ROOT,
    calculate_classification_metrics,
    encode_labels,
    load_params,
    make_class_mapping,
    project_path,
    read_dataset,
    save_json,
    set_random_seed,
    split_features_target,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def promote_base_model(paths: dict[str, Any]) -> None:
    base_model_path = project_path(paths.get("base_model_path", paths["model_path"]))
    base_legacy_model_path = project_path(
        paths.get("base_legacy_model_path", paths["legacy_model_path"])
    )
    model_path = project_path(paths["model_path"])
    legacy_model_path = project_path(paths["legacy_model_path"])

    if not base_model_path.exists() or not base_legacy_model_path.exists():
        raise FileNotFoundError("Base model artifacts are missing. Run src/model.py first.")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_model_path, model_path)
    shutil.copy2(base_legacy_model_path, legacy_model_path)


def compile_loaded_model(model: keras.Model, params: dict[str, Any]) -> keras.Model:
    model_params = params["model"]
    optimizer = keras.optimizers.Adam(learning_rate=float(model_params.get("learning_rate", 0.001)))
    model.compile(
        optimizer=optimizer,
        loss=model_params.get("loss", "sparse_categorical_crossentropy"),
        metrics=["accuracy"],
    )
    return model


def class_mapping_from_feature_info(feature_info: dict[str, Any]) -> tuple[list[Any], dict[Any, int]]:
    class_labels = feature_info["class_labels"]
    class_to_index = {label: index for index, label in enumerate(class_labels)}
    return class_labels, class_to_index


def make_callbacks(params: dict[str, Any]) -> list[keras.callbacks.Callback]:
    training_params = params["training"]
    early_stopping_params = training_params.get("early_stopping", {})
    if not early_stopping_params.get("enabled", True):
        return []

    return [
        keras.callbacks.EarlyStopping(
            monitor=early_stopping_params.get("monitor", "val_loss"),
            patience=int(early_stopping_params.get("patience", 20)),
            restore_best_weights=bool(early_stopping_params.get("restore_best_weights", True)),
        )
    ]


def main() -> None:
    params = load_params()
    paths = params["paths"]
    target_column = params["data"]["target_column"]
    training_params = params["training"]
    retrain_params = params.get("retrain", {})
    mode = retrain_params.get("mode", "fine_tune")

    if mode != "fine_tune":
        raise ValueError(
            "retrain.mode must be 'fine_tune'. Full retraining needs a separate "
            "DVC stage because it replaces upstream preprocessing artifacts."
        )

    seed = int(training_params.get("random_state", 42))
    set_random_seed(seed)

    flags_path = project_path(paths["retraining_flags_path"])
    if not flags_path.exists():
        raise FileNotFoundError(
            f"Retraining flags not found: {flags_path}. Run src/monitor.py before retraining."
        )

    flags = load_json(flags_path)
    if not flags.get("retraining_needed", False):
        promote_base_model(paths)
        checked_at = datetime.now(timezone.utc).isoformat()
        model_version = f"base-{checked_at.replace(':', '').replace('+', 'Z')}"
        blocked = bool(flags.get("retraining_blocked", False))
        metadata = {
            "created_at": checked_at,
            "status": "blocked" if blocked else "skipped",
            "reason": (
                "Incoming data failed quality checks."
                if blocked
                else "Monitoring did not request retraining."
            ),
            "model_version": model_version,
            "monitoring_flags": flags,
        }
        save_json(project_path(paths["retrain_metrics_path"]), metadata)
        save_json(project_path(paths["retrain_history_path"]), {})
        save_json(project_path(paths["retrain_metadata_path"]), metadata)
        save_text(project_path(paths["last_retrain_path"]), f"skipped:{checked_at}")
        save_text(project_path(paths["model_version_path"]), model_version)
        print(f"Retraining {metadata['status']}: {metadata['reason']}")
        return

    combined_train_path = project_path(paths["combined_train_data"])
    if not combined_train_path.exists():
        raise FileNotFoundError(
            f"Merged training data not found: {combined_train_path}. "
            "Run src/preprocess_new_data.py before retraining."
        )

    train_df = read_dataset(combined_train_path, target_column)
    test_df = read_dataset(project_path(paths["test_data"]), target_column)

    feature_info_path = project_path(paths["feature_info_path"])
    if feature_info_path.exists():
        feature_info = load_json(feature_info_path)
        feature_columns = feature_info["feature_columns"]
        class_labels, class_to_index = class_mapping_from_feature_info(feature_info)
    else:
        feature_info = {}
        feature_columns = None
        class_labels = []
        class_to_index = {}

    X_train_raw, y_train_labels, feature_columns = split_features_target(
        train_df, target_column, feature_columns
    )
    X_test_raw, y_test_labels, _ = split_features_target(test_df, target_column, feature_columns)

    if not class_to_index:
        class_labels, class_to_index = make_class_mapping(y_train_labels, y_test_labels)

    y_train = encode_labels(y_train_labels, class_to_index)
    y_test = encode_labels(y_test_labels, class_to_index)

    scaler_path = project_path(paths["scaler_path"])
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {scaler_path}. Run src/model.py before fine-tuning."
        )
    preprocessor = joblib.load(scaler_path)
    X_train = preprocessor.transform(X_train_raw).astype(np.float32)
    X_test = preprocessor.transform(X_test_raw).astype(np.float32)
    source_model_path = project_path(paths.get("base_model_path", paths["model_path"]))
    model = tf.keras.models.load_model(source_model_path, compile=False)
    model = compile_loaded_model(model, params)

    epochs = int(retrain_params.get("epochs", training_params.get("epochs", 100)))
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=int(training_params.get("batch_size", 32)),
        validation_split=float(training_params.get("validation_split", 0.2)),
        callbacks=make_callbacks(params),
        verbose=int(training_params.get("verbose", 1)),
    )

    train_probabilities = model.predict(X_train, verbose=0)
    test_probabilities = model.predict(X_test, verbose=0)
    y_train_pred = np.argmax(train_probabilities, axis=1)
    y_test_pred = np.argmax(test_probabilities, axis=1)
    train_metrics = calculate_classification_metrics(y_train, y_train_pred)
    test_metrics = calculate_classification_metrics(y_test, y_test_pred)

    model_path = project_path(paths["model_path"])
    legacy_model_path = project_path(paths["legacy_model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    model.save(legacy_model_path)

    history_payload = {
        metric: [float(value) for value in values] for metric, values in history.history.items()
    }
    save_json(project_path(paths["retrain_history_path"]), history_payload)
    save_json(
        project_path(paths["retrain_metrics_path"]),
        {
            "mode": mode,
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "classes": len(class_labels),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        },
    )

    retrained_at = datetime.now(timezone.utc).isoformat()
    model_version = retrained_at.replace(":", "").replace("+", "Z")
    metadata = {
        "created_at": retrained_at,
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "pandas_version": pd.__version__,
        "random_state": seed,
        "mode": mode,
        "epochs": epochs,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "input_dim": int(X_train.shape[1]),
        "num_classes": int(len(class_labels)),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "scaler_path": str(scaler_path.relative_to(PROJECT_ROOT)),
        "combined_train_data": paths["combined_train_data"],
        "model_version": model_version,
    }
    save_json(project_path(paths["retrain_metadata_path"]), metadata)
    save_text(project_path(paths["last_retrain_path"]), retrained_at)
    save_text(project_path(paths["model_version_path"]), model_version)

    print("Classification retraining complete")
    print(json.dumps({"mode": mode, "train_metrics": train_metrics, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
