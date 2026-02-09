#!/usr/bin/env python3
import argparse
import random
from collections import defaultdict
from math import sqrt

import numpy as np
import pm4py
from pm4py.util import xes_constants
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from prefix_feature_extraction import build_prefix_features_remaining_time
from time_transf import transform_time, inverse_transform_time


def select_components_by_cumulative_variance(
    explained_variance_ratio, threshold=0.93
):
    cumulative = 0.0
    for idx, ratio in enumerate(explained_variance_ratio):
        cumulative += ratio
        if cumulative >= threshold:
            return idx + 1
    return len(explained_variance_ratio)


def fit_pca_with_variance_threshold(features, threshold=0.93):
    if not features:
        raise ValueError("No features provided for PCA fitting.")
    max_components = min(len(features), len(features[0]))
    if max_components <= 1:
        pca = PCA(n_components=1)
        pca.fit(features)
        return 1, pca
    pca_full = PCA(n_components=max_components, random_state=42)
    pca_full.fit(features)
    n_components = select_components_by_cumulative_variance(
        pca_full.explained_variance_ratio_.tolist(), threshold=threshold
    )
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(features)
    return n_components, pca


def fit_pca_fixed_components(features, requested_components=3):
    if not features:
        raise ValueError("No features provided for PCA fitting.")
    max_components = min(len(features), len(features[0]))
    n_components = min(requested_components, max_components)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(features)
    return n_components, pca


def sample_training_data(features, targets, percentage, rng):
    if percentage >= 100:
        return features, targets
    sample_size = max(2, int(round(len(features) * (percentage / 100.0))))
    sample_size = min(sample_size, len(features))
    indices = rng.sample(range(len(features)), sample_size)
    sampled_features = [features[i] for i in indices]
    sampled_targets = [targets[i] for i in indices]
    return sampled_features, sampled_targets


def train_regressor(features, targets):
    reg = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    )
    reg.fit(features, targets)
    return reg


def train_pca_knn_regressor(features, targets):
    n_components, pca = fit_pca_with_variance_threshold(features, threshold=0.93)
    reduced = pca.transform(features)
    reg = KNeighborsRegressor(n_neighbors=1)
    reg.fit(reduced, targets)
    return {"pca": pca, "model": reg, "n_components": n_components}


def train_pca_random_forest_regressor(features, targets, requested_components=3):
    n_components, pca = fit_pca_fixed_components(
        features, requested_components=requested_components
    )
    reduced = pca.transform(features)
    reg = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    )
    reg.fit(reduced, targets)
    return {"pca": pca, "model": reg, "n_components": n_components}


def train_ridge_regressor(features, targets):
    reg = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=50.0)),
        ]
    )
    reg.fit(features, targets)
    return reg


def train_hgbr_regressor(features, targets):
    reg = HistGradientBoostingRegressor(
        random_state=42,
        max_depth=3,
        learning_rate=0.05,
        max_iter=400,
        min_samples_leaf=10,
        l2_regularization=2.0,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
    )
    reg.fit(features, targets)
    return reg


def _to_hours(values):
    hours = inverse_transform_time(np.asarray(values, dtype=float))
    return np.maximum(hours, 0.0)


def compute_regression_metrics(targets_transformed, predictions_transformed, case_ids):
    targets_hours = _to_hours(targets_transformed)
    preds_hours = _to_hours(predictions_transformed)

    abs_errors = [abs(y_true - y_hat) for y_true, y_hat in zip(targets_hours, preds_hours)]
    sq_errors = [(y_true - y_hat) ** 2 for y_true, y_hat in zip(targets_hours, preds_hours)]

    per_case_abs = defaultdict(list)
    per_case_sq = defaultdict(list)
    for case_id, ae, se in zip(case_ids, abs_errors, sq_errors):
        per_case_abs[case_id].append(ae)
        per_case_sq[case_id].append(se)

    case_mae_values = [sum(vals) / len(vals) for vals in per_case_abs.values()]
    case_rmse_values = [sqrt(sum(vals) / len(vals)) for vals in per_case_sq.values()]

    mae = sum(case_mae_values) / len(case_mae_values)
    rmse = sum(case_rmse_values) / len(case_rmse_values)

    return {
        "mae_hours": mae,
        "rmse_hours": rmse,
        "r2": r2_score(targets_hours, preds_hours),
    }


def evaluate_regressor(model, features, targets, case_ids):
    predictions = model.predict(features)
    return compute_regression_metrics(targets, predictions, case_ids)


def evaluate_pca_knn_regressor(model_bundle, features, targets, case_ids):
    reduced = model_bundle["pca"].transform(features)
    predictions = model_bundle["model"].predict(reduced)
    return compute_regression_metrics(targets, predictions, case_ids)


def evaluate_pca_random_forest_regressor(model_bundle, features, targets, case_ids):
    reduced = model_bundle["pca"].transform(features)
    predictions = model_bundle["model"].predict(reduced)
    return compute_regression_metrics(targets, predictions, case_ids)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Encode every prefix of each trace using the activities and paths up to the "
            "penultimate event, plus time-based features, with remaining time to case end as target."
        )
    )
    parser.add_argument(
        "log_path",
        nargs="?",
        default="C:/receipt.xes",
        help="Path to the XES log (default: tests/input_data/receipt.xes)",
    )
    parser.add_argument(
        "--activity-key",
        default=xes_constants.DEFAULT_NAME_KEY,
        help=f"Event attribute to use as activity (default: {xes_constants.DEFAULT_NAME_KEY})",
    )
    parser.add_argument(
        "--timestamp-key",
        default=xes_constants.DEFAULT_TIMESTAMP_KEY,
        help=f"Event attribute to use as timestamp (default: {xes_constants.DEFAULT_TIMESTAMP_KEY})",
    )
    parser.add_argument(
        "--show-sample",
        action="store_true",
        help="Print the first 5 feature rows and targets",
    )
    args = parser.parse_args()

    log = pm4py.read_xes(args.log_path, return_legacy_log_object=True)
    (
        feature,
        target,
        case_ids,
        activities,
        activity_to_index,
        paths,
        _,
    ) = build_prefix_features_remaining_time(
        log, args.activity_key, args.timestamp_key
    )
    if not feature:
        raise SystemExit("No prefixes with timestamps found in the log.")

    candidate_percentages = [0.5, 1, 3, 5]

    targets_hours = np.maximum(np.asarray(target, dtype=float) / 3600.0, 0.0)
    targets_transformed = transform_time(targets_hours).tolist()

    X_train, X_test, y_train, y_test, _, case_test = train_test_split(
        feature, targets_transformed, case_ids, test_size=0.2, random_state=42
    )

    print(f"Log path: {args.log_path}")
    print(f"Activity key: {args.activity_key}")
    print(f"Timestamp key: {args.timestamp_key}")
    print(f"Activities: {len(activities)}")
    print(f"Paths: {len(paths)}")
    print(f"Samples (prefixes): {len(feature)}")
    print(f"Feature dimension: {len(activities) + len(paths) + 3}")
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    rng = random.Random(42)
    for percentage in candidate_percentages:
        X_sampled, y_sampled = sample_training_data(
            X_train, y_train, percentage, rng
        )
        reg = train_regressor(X_sampled, y_sampled)
        metrics = evaluate_regressor(reg, X_test, y_test, case_test)
        pca_rf = train_pca_random_forest_regressor(
            X_sampled, y_sampled, requested_components=3
        )
        pca_rf_metrics = evaluate_pca_random_forest_regressor(
            pca_rf, X_test, y_test, case_test
        )
        pca_knn = train_pca_knn_regressor(X_sampled, y_sampled)
        pca_metrics = evaluate_pca_knn_regressor(
            pca_knn, X_test, y_test, case_test
        )
        ridge = train_ridge_regressor(X_sampled, y_sampled)
        ridge_metrics = evaluate_regressor(ridge, X_test, y_test, case_test)
        hgbr = train_hgbr_regressor(X_sampled, y_sampled)
        hgbr_metrics = evaluate_regressor(hgbr, X_test, y_test, case_test)
        print(f"Training sample %: {percentage}")
        print(f"Train size (sampled): {len(X_sampled)}")
        print(f"RF (no PCA) Per-case MAE (hours): {metrics['mae_hours']:.4f}")
        print(f"RF (no PCA) Per-case RMSE (hours): {metrics['rmse_hours']:.4f}")
        print(f"RF (no PCA) R2: {metrics['r2']:.4f}")
        print(
            "StandardScaler+Ridge "
            f"MAE (hours): {ridge_metrics['mae_hours']:.4f} "
            f"RMSE (hours): {ridge_metrics['rmse_hours']:.4f} "
            f"R2: {ridge_metrics['r2']:.4f}"
        )
        print(
            "HistGradientBoostingRegressor "
            f"MAE (hours): {hgbr_metrics['mae_hours']:.4f} "
            f"RMSE (hours): {hgbr_metrics['rmse_hours']:.4f} "
            f"R2: {hgbr_metrics['r2']:.4f}"
        )
        print(
            "PCA(3)+RF "
            f"MAE (hours): {pca_rf_metrics['mae_hours']:.4f} "
            f"RMSE (hours): {pca_rf_metrics['rmse_hours']:.4f} "
            f"R2: {pca_rf_metrics['r2']:.4f} "
            f"(components: {pca_rf['n_components']})"
        )
        print(
            "PCA+kNN (k=1) "
            f"MAE (hours): {pca_metrics['mae_hours']:.4f} "
            f"RMSE (hours): {pca_metrics['rmse_hours']:.4f} "
            f"R2: {pca_metrics['r2']:.4f} "
            f"(components: {pca_knn['n_components']})"
        )

    if args.show_sample:
        print("Sample features (first 5 rows):")
        for row in feature[:5]:
            print(row)
        print("Sample targets (first 5):")
        print(target[:5])


if __name__ == "__main__":
    main()
