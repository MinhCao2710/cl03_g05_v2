"""Evaluate the trained classifier and create classification reports."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


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


def decode_labels(indices: np.ndarray, class_labels: list[Any]) -> list[Any]:
    return [class_labels[int(index)] for index in indices]


def save_confusion_matrix_plot(
    y_true: np.ndarray, y_pred: np.ndarray, class_labels: list[Any], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(class_labels)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels([str(label) for label in class_labels])
    ax.set_yticklabels([str(label) for label in class_labels])

    threshold = cm.max() / 2 if cm.size else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            color = "white" if cm[row, col] > threshold else "black"
            ax.text(col, row, int(cm[row, col]), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_prediction_distribution_plot(
    y_true: np.ndarray, y_pred: np.ndarray, class_labels: list[Any], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.arange(len(class_labels))
    actual_counts = np.bincount(y_true.astype(int), minlength=len(class_labels))
    predicted_counts = np.bincount(y_pred.astype(int), minlength=len(class_labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.38
    ax.bar(labels - width / 2, actual_counts, width=width, label="Actual")
    ax.bar(labels + width / 2, predicted_counts, width=width, label="Predicted")
    ax.set_title("Actual vs Predicted Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Rows")
    ax.set_xticks(labels)
    ax.set_xticklabels([str(label) for label in class_labels])
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    params = load_params()
    paths = params["paths"]
    feature_info = load_json(project_path(paths["feature_info_path"]))
    class_labels = feature_info["class_labels"]

    model = tf.keras.models.load_model(project_path(paths["model_path"]), compile=False)
    X_test = np.load(project_path(paths["x_test_path"]))
    y_test = np.load(project_path(paths["y_test_path"])).reshape(-1).astype(int)

    probabilities = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1).astype(int)
    confidence = np.max(probabilities, axis=1)

    metrics = calculate_classification_metrics(y_test, y_pred)
    metrics_payload = {"rows": int(len(y_test)), "classes": len(class_labels), **metrics}
    save_json(project_path(paths["evaluation_metrics_path"]), metrics_payload)

    report_payload = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(class_labels))),
        target_names=[str(label) for label in class_labels],
        output_dict=True,
        zero_division=0,
    )
    save_json(project_path(paths["classification_report_path"]), report_payload)

    predictions_df = pd.DataFrame(
        {
            "row_id": np.arange(len(y_test)),
            "actual": decode_labels(y_test, class_labels),
            "prediction": decode_labels(y_pred, class_labels),
            "correct": y_test == y_pred,
            "confidence": confidence,
        }
    )
    for class_index, class_label in enumerate(class_labels):
        predictions_df[f"prob_class_{class_label}"] = probabilities[:, class_index]

    predictions_path = project_path(paths["test_predictions_path"])
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(predictions_path, index=False)

    submission_df = pd.DataFrame(
        {"row_id": np.arange(len(y_pred)), "prediction": decode_labels(y_pred, class_labels)}
    )
    submission_path = project_path(paths["submission_path"])
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(submission_path, index=False)

    save_confusion_matrix_plot(
        y_test, y_pred, class_labels, project_path(paths["confusion_matrix_plot_path"])
    )
    save_prediction_distribution_plot(
        y_test, y_pred, class_labels, project_path(paths["prediction_distribution_plot_path"])
    )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": paths["model_path"],
        "x_test_path": paths["x_test_path"],
        "y_test_path": paths["y_test_path"],
        "metrics_path": paths["evaluation_metrics_path"],
        "classification_report_path": paths["classification_report_path"],
    }
    save_json(project_path(paths["evaluation_metadata_path"]), metadata)

    print("Classification evaluation complete")
    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
