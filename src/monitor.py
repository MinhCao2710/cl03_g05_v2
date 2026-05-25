"""Monitor classifier performance and decide whether retraining is needed."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_params() -> dict[str, Any]:
    with project_path("params.yaml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


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


def squeeze_model_input(X: np.ndarray) -> np.ndarray:
    if X.ndim == 3 and X.shape[-1] == 1:
        return X.reshape((X.shape[0], X.shape[1]))
    return X


def predict_classes(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    probabilities = model.predict(squeeze_model_input(X), verbose=0)
    return np.argmax(probabilities, axis=1).astype(int)


def main() -> None:
    params = load_params()
    paths = params["paths"]
    monitoring_params = params["monitoring"]

    model = tf.keras.models.load_model(project_path(paths["model_path"]), compile=False)

    X_test = np.load(project_path(paths["x_test_path"]))
    y_test = np.load(project_path(paths["y_test_path"])).reshape(-1).astype(int)
    y_test_pred = predict_classes(model, X_test)
    old_metrics = calculate_classification_metrics(y_test, y_test_pred)

    X_new = np.load(project_path(paths["x_new_path"]))
    y_new = np.load(project_path(paths["y_new_path"])).reshape(-1)
    labelled_mask = np.isfinite(y_new)

    new_metrics: dict[str, float] | None = None
    performance_change: dict[str, float] | None = None
    reasons: list[str] = []

    min_labelled_rows = int(monitoring_params.get("min_labelled_new_rows", 20))
    if int(labelled_mask.sum()) < min_labelled_rows:
        reasons.append(
            f"Only {int(labelled_mask.sum())} labelled new rows available; "
            f"{min_labelled_rows} required for a retraining decision."
        )
    else:
        X_new_labelled = X_new[labelled_mask]
        y_new_labelled = y_new[labelled_mask].astype(int)
        y_new_pred = predict_classes(model, X_new_labelled)
        new_metrics = calculate_classification_metrics(y_new_labelled, y_new_pred)
        performance_change = {
            "accuracy_absolute_change": float(new_metrics["accuracy"] - old_metrics["accuracy"]),
            "precision_absolute_change": float(new_metrics["precision"] - old_metrics["precision"]),
            "recall_absolute_change": float(new_metrics["recall"] - old_metrics["recall"]),
            "f1_absolute_change": float(new_metrics["f1"] - old_metrics["f1"]),
        }

        accuracy_drop_threshold = float(monitoring_params.get("accuracy_drop_threshold", 0.05))
        precision_drop_threshold = float(monitoring_params.get("precision_drop_threshold", 0.05))
        recall_drop_threshold = float(monitoring_params.get("recall_drop_threshold", 0.05))
        f1_drop_threshold = float(monitoring_params.get("f1_drop_threshold", 0.05))

        if performance_change["accuracy_absolute_change"] < -accuracy_drop_threshold:
            reasons.append(
                f"Accuracy dropped by {abs(performance_change['accuracy_absolute_change']):.4f}, "
                f"above threshold {accuracy_drop_threshold:.4f}."
            )
        if performance_change["precision_absolute_change"] < -precision_drop_threshold:
            reasons.append(
                f"Precision dropped by {abs(performance_change['precision_absolute_change']):.4f}, "
                f"above threshold {precision_drop_threshold:.4f}."
            )
        if performance_change["recall_absolute_change"] < -recall_drop_threshold:
            reasons.append(
                f"Recall dropped by {abs(performance_change['recall_absolute_change']):.4f}, "
                f"above threshold {recall_drop_threshold:.4f}."
            )
        if performance_change["f1_absolute_change"] < -f1_drop_threshold:
            reasons.append(
                f"F1 dropped by {abs(performance_change['f1_absolute_change']):.4f}, "
                f"above threshold {f1_drop_threshold:.4f}."
            )

    retraining_needed = bool(
        new_metrics is not None
        and any(
            reason.startswith("Accuracy")
            or reason.startswith("Precision")
            or reason.startswith("Recall")
            or reason.startswith("F1")
            for reason in reasons
        )
    )

    flags = {
        "retraining_needed": retraining_needed,
        "labelled_new_rows": int(labelled_mask.sum()),
        "min_labelled_new_rows": min_labelled_rows,
        "reason_count": len(reasons),
    }
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "old_test_metrics": old_metrics,
        "new_data_metrics": new_metrics,
        "performance_change": performance_change,
        "flags": flags,
        "reasons": reasons,
    }

    save_json(project_path(paths["monitoring_metrics_path"]), summary)
    save_json(project_path(paths["monitoring_summary_path"]), summary)
    save_json(project_path(paths["retraining_flags_path"]), flags)

    print("Classification monitoring complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
