"""Monitor classifier performance and decide whether retraining is needed."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# valid value ranges for this dataset 
FEATURE_VALID_RANGES: dict[str, tuple[float, float]] = {
    "BMI": (10.0, 70.0),
    "Eating_risk": (-4.0, 4.0),
    "Activity_risk": (-4.0, 4.0),
    "Lifestyle_risk": (-1.0, 10.0),
    "family_history_with_overweight": (0.0, 1.0),
}

VALID_LABEL_VALUES: set[int] = {0, 1, 2, 3, 4, 5, 6}

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


def calculate_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def squeeze_model_input(X: np.ndarray) -> np.ndarray:
    if X.ndim == 3 and X.shape[-1] == 1:
        return X.reshape((X.shape[0], X.shape[1]))
    return X


def predict_classes(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    probabilities = model.predict(squeeze_model_input(X), verbose=0)
    return np.argmax(probabilities, axis=1).astype(int)


# data quality checks
def check_data_quality(
    new_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    valid_label_values: set[int],
) -> dict[str, Any]:
    # it run all data quality checks on the incoming new_data DataFrame.
    # we checks 
    #     1 missing feature columns
    #     2 extra / unexpected columns
    #     3 null / NaN values per column
    #     4 duplicate rows
    #     5 invalid value ranges
    #     6 unknown label values (if target present)
    issues: list[str] = []
    details: dict[str, Any] = {}

    if new_df.empty:
        issues.append("New data file contains no rows.")

    # 1. Missing feature columns
    missing_cols = [c for c in feature_columns if c not in new_df.columns]
    details["missing_feature_columns"] = missing_cols
    if missing_cols:
        issues.append(
            f"Missing feature columns: {missing_cols}. "
            "Affected rows will be imputed with NaN."
        )

    # 2. Extra / unexpected columns (excluding the target column)
    expected = set(feature_columns) | {target_column}
    extra_cols = [c for c in new_df.columns if c not in expected]
    details["extra_columns"] = extra_cols
    if extra_cols:
        issues.append(f"Unexpected extra columns found (will be ignored): {extra_cols}")

    # 3. Null / NaN values
    null_counts = new_df.reindex(columns=feature_columns).isnull().sum()
    null_report = {col: int(cnt) for col, cnt in null_counts.items() if cnt > 0}
    details["null_counts_per_column"] = null_report
    total_nulls = sum(null_report.values())
    if null_report:
        issues.append(
            f"Null values detected in {len(null_report)} column(s), "
            f"{total_nulls} total null cells: {null_report}"
        )

    # Null rate per column (fraction)
    null_rates = {
        col: round(int(cnt) / len(new_df), 4) if len(new_df) else 0.0
        for col, cnt in null_counts.items()
        if col in feature_columns
    }
    details["null_rates_per_column"] = null_rates

    # 4. Duplicate rows
    dup_count = int(new_df.duplicated().sum())
    details["duplicate_rows"] = dup_count
    if dup_count > 0:
        issues.append(
            f"{dup_count} duplicate row(s) found in new data."
        )

    # 5. Invalid value ranges
    out_of_range: dict[str, dict[str, Any]] = {}
    for col, (lo, hi) in FEATURE_VALID_RANGES.items():
        if col not in new_df.columns:
            continue
        series = pd.to_numeric(new_df[col], errors="coerce")
        bad_mask = series.notna() & ((series < lo) | (series > hi))
        bad_count = int(bad_mask.sum())
        if bad_count > 0:
            out_of_range[col] = {
                "expected_range": [lo, hi],
                "out_of_range_count": bad_count,
                "sample_values": series[bad_mask].head(5).tolist(),
            }
    details["out_of_range_values"] = out_of_range
    if out_of_range:
        for col, info in out_of_range.items():
            issues.append(
                f"Column '{col}': {info['out_of_range_count']} value(s) "
                f"outside expected range {info['expected_range']}."
            )

    # 6. Unknown label values (only if target column present)
    unknown_labels: list[Any] = []
    if target_column in new_df.columns:
        label_series = pd.to_numeric(new_df[target_column], errors="coerce").dropna()
        unique_labels = set(label_series.unique())
        unknown_labels = sorted(
            value for value in unique_labels
            if not float(value).is_integer() or int(value) not in valid_label_values
        )
        details["unknown_label_values"] = unknown_labels
        if unknown_labels:
            issues.append(
                f"Unknown label value(s) in '{target_column}': {unknown_labels}. "
                f"Valid labels are {sorted(valid_label_values)}."
            )
    else:
        details["unknown_label_values"] = []

    passed = len(issues) == 0
    return {
        "passed": passed,
        "issue_count": len(issues),
        "issues": issues,
        "details": details,
    }


# drift checks 
def check_drift(
    train_df: pd.DataFrame,
    new_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    mean_shift_threshold: float = 0.20,
    std_shift_threshold: float = 0.30,
    ks_pvalue_threshold: float = 0.05,
    class_dist_threshold: float = 0.10,
    missing_rate_threshold: float = 0.05,
) -> dict[str, Any]:
    # detect distribution shift between training data and new incoming data.

    # Checks:
    #     1. Feature mean shift (relative change)
    #     2. Feature std shift (relative change)
    #     3. KS-test p-value per feature (statistical significance)
    #     4. Class distribution shift (per-class fraction change)
    #     5. Missing-rate shift per feature
    
    drift_flags: list[str] = []
    details: dict[str, Any] = {}

    feature_drift: dict[str, Any] = {}
    for col in feature_columns:
        if col not in train_df.columns or col not in new_df.columns:
            continue

        train_series = pd.to_numeric(train_df[col], errors="coerce").dropna()
        new_series = pd.to_numeric(new_df[col], errors="coerce").dropna()

        if len(train_series) == 0 or len(new_series) == 0:
            continue

        train_mean = float(train_series.mean())
        new_mean = float(new_series.mean())
        train_std = float(train_series.std())
        new_std = float(new_series.std())

        # Relative mean shift (avoid div-by-zero)
        if train_mean != 0:
            mean_rel_change = abs(new_mean - train_mean) / abs(train_mean)
        else:
            mean_rel_change = abs(new_mean - train_mean)

        # Relative std shift
        if train_std != 0:
            std_rel_change = abs(new_std - train_std) / abs(train_std)
        else:
            std_rel_change = abs(new_std - train_std)

        # KS-test
        ks_stat, ks_pvalue = stats.ks_2samp(train_series.values, new_series.values)

        col_info: dict[str, Any] = {
            "train_mean": round(train_mean, 6),
            "new_mean": round(new_mean, 6),
            "mean_relative_change": round(mean_rel_change, 6),
            "mean_drifted": mean_rel_change > mean_shift_threshold,
            "train_std": round(train_std, 6),
            "new_std": round(new_std, 6),
            "std_relative_change": round(std_rel_change, 6),
            "std_drifted": std_rel_change > std_shift_threshold,
            "ks_statistic": round(float(ks_stat), 6),
            "ks_pvalue": round(float(ks_pvalue), 6),
            "ks_significant_drift": float(ks_pvalue) < ks_pvalue_threshold,
        }
        feature_drift[col] = col_info

        if col_info["mean_drifted"]:
            drift_flags.append(
                f"Feature '{col}': mean shifted by "
                f"{mean_rel_change:.1%} (threshold {mean_shift_threshold:.0%})."
            )
        if col_info["std_drifted"]:
            drift_flags.append(
                f"Feature '{col}': std shifted by "
                f"{std_rel_change:.1%} (threshold {std_shift_threshold:.0%})."
            )
        if col_info["ks_significant_drift"]:
            drift_flags.append(
                f"Feature '{col}': KS-test p-value={ks_pvalue:.4f} "
                f"< {ks_pvalue_threshold} — statistically significant distribution shift."
            )

    details["feature_drift"] = feature_drift

    # Class distribution shift
    class_dist_result: dict[str, Any] = {}
    if target_column in train_df.columns and target_column in new_df.columns:
        train_dist = (
            pd.to_numeric(train_df[target_column], errors="coerce")
            .value_counts(normalize=True)
            .to_dict()
        )
        new_dist = (
            pd.to_numeric(new_df[target_column], errors="coerce")
            .value_counts(normalize=True)
            .to_dict()
        )
        all_classes = set(train_dist) | set(new_dist)
        per_class: dict[str, Any] = {}
        for cls in sorted(all_classes):
            train_frac = float(train_dist.get(cls, 0.0))
            new_frac = float(new_dist.get(cls, 0.0))
            change = abs(new_frac - train_frac)
            drifted = change > class_dist_threshold
            per_class[str(int(cls))] = {
                "train_fraction": round(train_frac, 6),
                "new_fraction": round(new_frac, 6),
                "absolute_change": round(change, 6),
                "drifted": drifted,
            }
            if drifted:
                drift_flags.append(
                    f"Class '{int(cls)}' distribution changed by "
                    f"{change:.1%} (threshold {class_dist_threshold:.0%})."
                )
        class_dist_result = {"per_class": per_class}
    details["class_distribution_drift"] = class_dist_result

    # Missing-rate shift per feature
    missing_rate_drift: dict[str, Any] = {}
    for col in feature_columns:
        if col not in train_df.columns or col not in new_df.columns:
            continue
        train_miss_rate = float(train_df[col].isnull().mean())
        new_miss_rate = float(new_df[col].isnull().mean())
        change = abs(new_miss_rate - train_miss_rate)
        drifted = change > missing_rate_threshold
        missing_rate_drift[col] = {
            "train_missing_rate": round(train_miss_rate, 6),
            "new_missing_rate": round(new_miss_rate, 6),
            "absolute_change": round(change, 6),
            "drifted": drifted,
        }
        if drifted:
            drift_flags.append(
                f"Feature '{col}': missing rate changed from "
                f"{train_miss_rate:.1%} to {new_miss_rate:.1%} "
                f"(change {change:.1%} > threshold {missing_rate_threshold:.0%})."
            )
    details["missing_rate_drift"] = missing_rate_drift

    drift_detected = len(drift_flags) > 0
    return {
        "drift_detected": drift_detected,
        "drift_flag_count": len(drift_flags),
        "drift_flags": drift_flags,
        "details": details,
    }



def check_performance(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_new: np.ndarray,
    y_new: np.ndarray,
    monitoring_params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    """Compare model performance on test vs new labelled data."""
    y_test_pred = predict_classes(model, X_test)
    old_metrics = calculate_classification_metrics(y_test, y_test_pred)

    reasons: list[str] = []
    new_metrics: dict[str, Any] | None = None
    performance_change: dict[str, Any] | None = None

    min_labelled = int(monitoring_params.get("min_labelled_new_rows", 20))
    labelled_mask = np.isfinite(y_new)
    labelled_count = int(labelled_mask.sum())

    if labelled_count < min_labelled:
        reasons.append(
            f"Only {labelled_count} labelled new rows available; "
            f"{min_labelled} required for performance comparison."
        )
        return old_metrics, new_metrics, performance_change, reasons

    X_new_labelled = X_new[labelled_mask]
    y_new_labelled = y_new[labelled_mask].astype(int)
    y_new_pred = predict_classes(model, X_new_labelled)
    new_metrics = calculate_classification_metrics(y_new_labelled, y_new_pred)

    accuracy_threshold = float(monitoring_params.get("accuracy_drop_threshold", 0.05))
    precision_threshold = float(monitoring_params.get("precision_drop_threshold", 0.05))
    recall_threshold = float(monitoring_params.get("recall_drop_threshold", 0.05))
    f1_threshold = float(monitoring_params.get("f1_drop_threshold", 0.05))

    performance_change = {
        "accuracy_absolute_change": round(
            new_metrics["accuracy"] - old_metrics["accuracy"], 6
        ),
        "precision_absolute_change": round(
            new_metrics["precision"] - old_metrics["precision"], 6
        ),
        "recall_absolute_change": round(
            new_metrics["recall"] - old_metrics["recall"], 6
        ),
        "f1_absolute_change": round(new_metrics["f1"] - old_metrics["f1"], 6),
    }

    if performance_change["accuracy_absolute_change"] < -accuracy_threshold:
        reasons.append(
            f"Accuracy dropped by "
            f"{abs(performance_change['accuracy_absolute_change']):.4f}, "
            f"exceeds threshold {accuracy_threshold}."
        )
    if performance_change["precision_absolute_change"] < -precision_threshold:
        reasons.append(
            f"Precision dropped by "
            f"{abs(performance_change['precision_absolute_change']):.4f}, "
            f"exceeds threshold {precision_threshold}."
        )
    if performance_change["recall_absolute_change"] < -recall_threshold:
        reasons.append(
            f"Recall dropped by "
            f"{abs(performance_change['recall_absolute_change']):.4f}, "
            f"exceeds threshold {recall_threshold}."
        )
    if performance_change["f1_absolute_change"] < -f1_threshold:
        reasons.append(
            f"F1 dropped by "
            f"{abs(performance_change['f1_absolute_change']):.4f}, "
            f"exceeds threshold {f1_threshold}."
        )

    return old_metrics, new_metrics, performance_change, reasons




def main() -> None:
    params = load_params()
    paths = params["paths"]
    monitoring_params = params["monitoring"]

    # Load model and arrays
    monitor_model_path = paths.get("monitor_model_path", paths.get("base_model_path", paths["model_path"]))
    model = tf.keras.models.load_model(project_path(monitor_model_path), compile=False)
    X_test = np.load(project_path(paths["x_test_path"]))
    y_test = np.load(project_path(paths["y_test_path"])).reshape(-1).astype(int)
    X_new = np.load(project_path(paths["x_new_path"]))
    y_new = np.load(project_path(paths["y_new_path"])).reshape(-1)

    # Load raw CSVs for quality/drift checks
    train_df = pd.read_csv(project_path(paths["train_data"]))
    new_df = pd.read_csv(project_path(paths["new_data"]))

    target_column = params["data"]["target_column"]
    feature_columns = [c for c in train_df.columns if c != target_column]
    feature_info_path = project_path(paths["feature_info_path"])
    if feature_info_path.exists():
        with feature_info_path.open("r", encoding="utf-8") as file:
            feature_info = json.load(file)
        valid_label_values = {int(label) for label in feature_info.get("class_labels", VALID_LABEL_VALUES)}
    else:
        valid_label_values = VALID_LABEL_VALUES

    # ── Run checks ──
    quality_report = check_data_quality(new_df, feature_columns, target_column, valid_label_values)

    drift_report = check_drift(
        train_df,
        new_df,
        feature_columns,
        target_column,
        mean_shift_threshold=float(monitoring_params.get("mean_shift_threshold", 0.20)),
        std_shift_threshold=float(monitoring_params.get("std_shift_threshold", 0.30)),
        ks_pvalue_threshold=float(monitoring_params.get("ks_pvalue_threshold", 0.05)),
        class_dist_threshold=float(monitoring_params.get("class_dist_threshold", 0.10)),
        missing_rate_threshold=float(monitoring_params.get("missing_rate_threshold", 0.05)),
    )

    old_metrics, new_metrics, performance_change, perf_reasons = check_performance(
        model, X_test, y_test, X_new, y_new, monitoring_params
    )

    # ── Retraining decision ──
    performance_triggered = any(
        r.startswith(("Accuracy", "Precision", "Recall", "F1"))
        for r in perf_reasons
    )
    retraining_recommended = bool(
        new_metrics is not None and performance_triggered
    ) or drift_report["drift_detected"]
    retraining_blocked = not quality_report["passed"]
    retraining_needed = retraining_recommended and not retraining_blocked

    all_reasons: list[str] = perf_reasons + drift_report["drift_flags"]
    if not quality_report["passed"]:
        all_reasons = quality_report["issues"] + all_reasons

    labelled_mask = np.isfinite(y_new)
    flags = {
        "retraining_needed": retraining_needed,
        "retraining_recommended": retraining_recommended,
        "retraining_blocked": retraining_blocked,
        "triggered_by_performance": bool(new_metrics is not None and performance_triggered),
        "triggered_by_drift": drift_report["drift_detected"],
        "triggered_by_data_quality": not quality_report["passed"],
        "labelled_new_rows": int(labelled_mask.sum()),
        "min_labelled_new_rows": int(monitoring_params.get("min_labelled_new_rows", 20)),
        "total_reason_count": len(all_reasons),
    }

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_quality": quality_report,
        "drift": drift_report,
        "old_test_metrics": old_metrics,
        "new_data_metrics": new_metrics,
        "performance_change": performance_change,
        "performance_reasons": perf_reasons,
        "flags": flags,
        "all_reasons": all_reasons,
    }

    save_json(project_path(paths["monitoring_metrics_path"]), summary)
    save_json(project_path(paths["monitoring_summary_path"]), summary)
    save_json(project_path(paths["retraining_flags_path"]), flags)

    # ── Console output ──
    print("=" * 60)
    print("MONITORING REPORT")
    print("=" * 60)

    print(f"\n[DATA QUALITY] {'PASS' if quality_report['passed'] else 'FAIL'}")
    if not quality_report["passed"]:
        for issue in quality_report["issues"]:
            print(f"  ⚠  {issue}")

    print(f"\n[DRIFT] {'DETECTED' if drift_report['drift_detected'] else 'NONE'}")
    if drift_report["drift_flags"]:
        for flag in drift_report["drift_flags"]:
            print(f"  ↕  {flag}")

    print(f"\n[PERFORMANCE]")
    print(f"  Old test  → accuracy={old_metrics['accuracy']:.4f}  f1={old_metrics['f1']:.4f}")
    if new_metrics:
        print(f"  New data  → accuracy={new_metrics['accuracy']:.4f}  f1={new_metrics['f1']:.4f}")
    if perf_reasons:
        for r in perf_reasons:
            print(f"  ✗  {r}")

    print(f"\n[RETRAINING NEEDED] {retraining_needed}")
    print("=" * 60)
    print(json.dumps(flags, indent=2))


if __name__ == "__main__":
    main()
