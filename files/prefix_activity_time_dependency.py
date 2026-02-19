#!/usr/bin/env python3
import argparse

import numpy as np
import pm4py
from pm4py.util import xes_constants
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from prefix_feature_extraction import build_prefix_features_next_activity


TIME_FEATURE_NAMES = [
    "prev_to_penultimate_sec",
    "start_to_penultimate_sec",
    "path_time_diff_sec",
]


def _distance_correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    if n < 2:
        return float("nan")
    ax = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    ay = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=2)
    ax = ax - ax.mean(axis=0, keepdims=True) - ax.mean(axis=1, keepdims=True) + ax.mean()
    ay = ay - ay.mean(axis=0, keepdims=True) - ay.mean(axis=1, keepdims=True) + ay.mean()
    dcov2 = np.mean(ax * ay)
    dvarx2 = np.mean(ax * ax)
    dvary2 = np.mean(ay * ay)
    denom = np.sqrt(max(dvarx2 * dvary2, 1e-12))
    if denom <= 0:
        return 0.0
    value = np.sqrt(max(dcov2, 0.0) / denom)
    return float(np.clip(value, 0.0, 1.0))


def _build_indicator_names(activities, paths):
    names = [f"A:{a}" for a in activities]
    names.extend([f"P:{a}->{b}" for (a, b) in paths])
    return names


def _subsample_indices(n, max_samples, rng):
    if n <= max_samples:
        return np.arange(n)
    idx = list(range(n))
    rng.shuffle(idx)
    return np.asarray(idx[:max_samples], dtype=int)


def _safe_auc_cv(x, y, cv_splits, n_jobs):
    if np.unique(y).size < 2:
        return None
    min_count = int(np.bincount(y).min())
    if min_count < 2:
        return None
    folds = min(cv_splits, min_count)
    if folds < 2:
        return None
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=n_jobs,
        class_weight="balanced_subsample",
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(model, x, y, cv=cv, scoring="roc_auc", n_jobs=n_jobs)
    except Exception:
        return None
    if scores.size == 0:
        return None
    return float(np.mean(scores))


def _safe_r2_cv(x, y, cv_splits, n_jobs):
    if np.unique(y).size < 2:
        return None
    folds = min(cv_splits, len(y))
    if folds < 2:
        return None
    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=n_jobs
    )
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(model, x, y, cv=cv, scoring="r2", n_jobs=n_jobs)
    except Exception:
        return None
    if scores.size == 0:
        return None
    return float(np.mean(scores))


def _report_mi_matrix(x_time, x_act, indicator_names, top_k):
    records = []
    mean_values = []
    weighted_values = []
    for i in range(x_act.shape[1]):
        y = x_act[:, i].astype(int)
        classes = np.unique(y)
        if classes.size < 2:
            continue
        prevalence = float(np.mean(y))
        weight = prevalence * (1.0 - prevalence)
        mi_vec = mutual_info_classif(
            x_time, y, discrete_features=False, random_state=42
        )
        for j, mi in enumerate(mi_vec):
            mi = float(mi)
            records.append((mi, indicator_names[i], TIME_FEATURE_NAMES[j], prevalence))
            mean_values.append(mi)
            weighted_values.append(mi * weight)
    if not records:
        print("MI(activity_indicators, time_features): skipped (no valid indicators)")
        return
    print(f"MI(activity_indicators, time_features) mean: {np.mean(mean_values):.6f}")
    total_w = sum([r[3] * (1.0 - r[3]) for r in records])
    if total_w > 0:
        print(
            "MI(activity_indicators, time_features) prevalence-weighted mean: "
            f"{sum(weighted_values) / total_w:.6f}"
        )
    top = sorted(records, key=lambda t: t[0], reverse=True)[:top_k]
    print(f"Top {len(top)} MI pairs:")
    for mi, ind_name, time_name, prevalence in top:
        print(
            f"  - {ind_name} vs {time_name}: MI={mi:.6f} "
            f"(prevalence={prevalence:.4f})"
        )


def _report_time_to_activity_predictability(
    x_time, x_act, cv_splits, n_jobs, max_indicators, rng
):
    valid_cols = []
    for i in range(x_act.shape[1]):
        y = x_act[:, i].astype(int)
        if np.unique(y).size < 2:
            continue
        p = float(np.mean(y))
        if p <= 0.0 or p >= 1.0:
            continue
        valid_cols.append((i, p))
    if not valid_cols:
        print("Predictability time -> activities: skipped (no valid indicators)")
        return

    valid_cols.sort(key=lambda t: t[1] * (1.0 - t[1]), reverse=True)
    if len(valid_cols) > max_indicators:
        valid_cols = valid_cols[:max_indicators]

    x_time_shuf = x_time.copy()
    rng.shuffle(x_time_shuf)

    auc_real = []
    auc_delta = []
    auc_weighted_real = []
    auc_weighted_delta = []
    eval_weights = []
    evaluated = 0
    for i, p in valid_cols:
        y = x_act[:, i].astype(int)
        real = _safe_auc_cv(x_time, y, cv_splits, n_jobs)
        if real is None:
            continue
        base = _safe_auc_cv(x_time_shuf, y, cv_splits, n_jobs)
        if base is None:
            continue
        w = p * (1.0 - p)
        auc_real.append(real)
        auc_delta.append(real - base)
        auc_weighted_real.append(real * w)
        auc_weighted_delta.append((real - base) * w)
        eval_weights.append(w)
        evaluated += 1
    if evaluated == 0:
        print("Predictability time -> activities: skipped (CV constraints)")
        return
    sum_w = sum(eval_weights)
    print(
        "Predictability time -> activities "
        f"(ROC-AUC, macro over indicators): {np.mean(auc_real):.4f}"
    )
    if sum_w > 0:
        print(
            "Predictability time -> activities "
            f"(ROC-AUC, prevalence-weighted): {sum(auc_weighted_real)/sum_w:.4f}"
        )
    print(
        "Permutation gap time -> activities "
        f"(AUC delta, macro): {np.mean(auc_delta):.4f}"
    )
    if sum_w > 0:
        print(
            "Permutation gap time -> activities "
            f"(AUC delta, prevalence-weighted): {sum(auc_weighted_delta)/sum_w:.4f}"
        )


def _report_activity_to_time_predictability(x_act, x_time, cv_splits, n_jobs, rng):
    print("Predictability activities -> time (R2 by time feature):")
    for j, time_name in enumerate(TIME_FEATURE_NAMES):
        y = x_time[:, j]
        real = _safe_r2_cv(x_act, y, cv_splits, n_jobs)
        if real is None:
            print(f"  - {time_name}: skipped")
            continue
        y_shuf = y.copy()
        rng.shuffle(y_shuf)
        base = _safe_r2_cv(x_act, y_shuf, cv_splits, n_jobs)
        if base is None:
            print(f"  - {time_name}: R2={real:.4f} | permutation gap=n/a")
            continue
        print(
            f"  - {time_name}: R2={real:.4f} | "
            f"permutation gap (delta R2)={real - base:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Dependency scorecard between activity/path indicators and time features "
            "in prefix feature vectors."
        )
    )
    parser.add_argument(
        "log_path",
        nargs="?",
        default="C:/roadtraffic_10000.xes.gz",
        help="Path to the XES log.",
    )
    parser.add_argument(
        "--activity-key",
        default=xes_constants.DEFAULT_NAME_KEY,
        help=f"Event activity attribute (default: {xes_constants.DEFAULT_NAME_KEY})",
    )
    parser.add_argument(
        "--timestamp-key",
        default=xes_constants.DEFAULT_TIMESTAMP_KEY,
        help=f"Event timestamp attribute (default: {xes_constants.DEFAULT_TIMESTAMP_KEY})",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3000,
        help="Maximum rows used for heavy metrics (distance correlation, CV).",
    )
    parser.add_argument(
        "--max-indicators",
        type=int,
        default=200,
        help="Maximum number of activity/path indicators for time->activity CV metrics.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Cross-validation splits for predictive metrics.",
    )
    parser.add_argument(
        "--top-k-mi",
        type=int,
        default=15,
        help="Number of top MI pairs to print.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for sklearn.",
    )
    args = parser.parse_args()

    log = pm4py.read_xes(args.log_path, return_legacy_log_object=True)
    features, _, activities, _, paths, _ = build_prefix_features_next_activity(
        log, args.activity_key, args.timestamp_key
    )
    if not features:
        raise SystemExit("No prefix features found in the log.")

    x = np.asarray(features, dtype=float)
    num_activity_cols = len(activities) + len(paths)
    if x.shape[1] < num_activity_cols + 3:
        raise SystemExit(
            "Unexpected feature shape: cannot split activity/path and time parts."
        )
    x_act = x[:, :num_activity_cols]
    x_time = x[:, num_activity_cols : num_activity_cols + 3]

    finite_mask = np.isfinite(x_time).all(axis=1)
    x_act = x_act[finite_mask]
    x_time = x_time[finite_mask]
    if x_act.shape[0] == 0:
        raise SystemExit("No finite rows left after filtering invalid time values.")

    rng = np.random.default_rng(42)
    sub_idx = _subsample_indices(x_act.shape[0], args.max_samples, rng)
    x_act_sub = x_act[sub_idx]
    x_time_sub = x_time[sub_idx]
    indicator_names = _build_indicator_names(activities, paths)

    print(f"Log path: {args.log_path}")
    print(f"Samples used: {x_act.shape[0]} (subsampled for heavy metrics: {len(sub_idx)})")
    print(f"Activity/path indicators: {x_act.shape[1]}")
    print(f"Time features: {x_time.shape[1]} ({', '.join(TIME_FEATURE_NAMES)})")

    dcor = _distance_correlation(x_act_sub, x_time_sub)
    print(f"Distance correlation(activity_block, time_block): {dcor:.6f}")

    _report_mi_matrix(x_time_sub, x_act_sub, indicator_names, args.top_k_mi)

    _report_time_to_activity_predictability(
        x_time_sub,
        x_act_sub,
        args.cv_splits,
        args.n_jobs,
        args.max_indicators,
        rng,
    )

    _report_activity_to_time_predictability(
        x_act_sub, x_time_sub, args.cv_splits, args.n_jobs, rng
    )


if __name__ == "__main__":
    main()
