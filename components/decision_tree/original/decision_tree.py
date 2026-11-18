"""PathSnap Decision Tree candidate classifier.

Reads trip folders containing .track, .nodes, .arcs and .route files.
Training labels are constructed by associating each GPS observation with the
geometrically nearest arc in that trip's reference .route sequence. This is a
proxy because the supplied files do not provide a GPS-row-to-route-row index.

The model scores candidate road arcs for each GPS observation. It is not, by
itself, the final sequence decoder; the team HMM/Viterbi stage should consume
these candidate probabilities.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             make_scorer, precision_score, recall_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedGroupKFold,
                                     cross_validate)
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

FEATURES = [
    "distance_m", "distance_start_m", "distance_end_m", "segment_length_m",
    "heading_difference_deg", "gps_speed_mps", "time_gap_s",
    "projection_fraction", "candidate_rank"
]
CV_METADATA = ["trip_id", "gps_index", "candidate_rank"]
ORIGINAL_PARAMS = {
    "criterion": "gini",
    "max_depth": 12,
    "min_samples_split": 20,
    "min_samples_leaf": 5,
    "max_features": None,
    "class_weight": "balanced",
    "ccp_alpha": 0.0,
    "random_state": 42,
}


class SearchDecisionTree(ClassifierMixin, BaseEstimator):
    """Keep candidate identifiers available to scorers, not to the classifier."""

    def __init__(
        self,
        criterion="gini",
        max_depth=12,
        min_samples_split=20,
        min_samples_leaf=5,
        max_features=None,
        class_weight="balanced",
        ccp_alpha=0.0,
        random_state=42,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state

    def fit(self, X, y):
        self.model_ = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
            random_state=self.random_state,
        )
        self.model_.fit(X.loc[:, FEATURES], y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X.loc[:, FEATURES])

    def predict_proba(self, X):
        return self.model_.predict_proba(X.loc[:, FEATURES])


def positive_class_probabilities(model, candidate_features):
    """Return P(proxy-positive) in candidate row order."""
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1:
        raise ValueError("The fitted model must contain the positive class (1).")
    missing = [name for name in FEATURES if name not in candidate_features.columns]
    if missing:
        raise ValueError(f"Candidate features are missing required columns: {missing}")
    return model.predict_proba(candidate_features.loc[:, FEATURES])[:, int(positive[0])]


def score_candidates(model_bundle, candidate_features):
    """Score candidates for HMM/Viterbi and preserve identifiers and row order."""
    required_ids = ["trip_id", "gps_index", "arc_id"]
    missing = [name for name in required_ids if name not in candidate_features.columns]
    if missing:
        raise ValueError(f"Candidate rows are missing required identifiers: {missing}")
    probabilities = positive_class_probabilities(model_bundle["model"], candidate_features)
    scores = candidate_features.loc[:, required_ids].copy()
    scores["candidate_probability"] = probabilities
    return scores


def top1_proxy_accuracy_scorer(estimator, X, y):
    """Score ranking on covered GPS points; resolve ties by exact-distance rank."""
    positive = np.flatnonzero(np.asarray(estimator.classes_) == 1)
    if len(positive) != 1:
        raise ValueError("The fitted estimator must contain positive class (1).")
    probabilities = estimator.predict_proba(X)[:, int(positive[0])]
    ranked = X.loc[:, CV_METADATA].copy()
    ranked["label"] = np.asarray(y, dtype=int)
    ranked["score"] = probabilities
    best = (
        ranked.sort_values(
            ["trip_id", "gps_index", "score", "candidate_rank"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .drop_duplicates(["trip_id", "gps_index"], keep="first")
    )
    return float(best["label"].mean()) if len(best) else 0.0


def sample_cv_points(frame, max_points_per_trip, random_state):
    """Sample whole GPS candidate sets while retaining every training trip."""
    rng = np.random.default_rng(random_state)
    sampled = []
    for _, trip_frame in frame.groupby("trip_id", sort=True):
        point_ids = trip_frame["gps_index"].unique()
        if len(point_ids) > max_points_per_trip:
            point_ids = rng.choice(point_ids, size=max_points_per_trip, replace=False)
        sampled.append(trip_frame.loc[trip_frame["gps_index"].isin(point_ids)])
    if not sampled:
        raise ValueError("No covered training observations are available for CV.")
    return pd.concat(sampled, ignore_index=True)


def evaluate_candidate_rows(frame, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    labels = frame["label"].to_numpy(dtype=int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix_labels_0_1": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }


def summarize_cv_results(results, metrics):
    summary = {}
    for metric in metrics:
        values = [float(value) for value in results[f"test_{metric}"]]
        summary[metric] = {
            "fold_scores": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    return summary


def summarize_search_cv(search, metrics):
    best_index = int(search.best_index_)
    summary = {}
    for metric in metrics:
        values = [
            float(search.cv_results_[f"split{fold}_test_{metric}"][best_index])
            for fold in range(search.n_splits_)
        ]
        summary[metric] = {
            "fold_scores": values,
            "mean": float(search.cv_results_[f"mean_test_{metric}"][best_index]),
            "std": float(search.cv_results_[f"std_test_{metric}"][best_index]),
        }
    return summary


def read_trip(folder: Path):
    trip_id = folder.name
    def load(suffix, dtype=float):
        p = folder / f"{trip_id}.{suffix}"
        if not p.exists():
            raise FileNotFoundError(p)
        return np.loadtxt(p, dtype=dtype, ndmin=2)
    track = load("track")
    nodes = load("nodes")
    arcs = load("arcs", int).astype(int)
    route = np.loadtxt(folder / f"{trip_id}.route", dtype=int, ndmin=1)
    if track.shape[1] != 3 or nodes.shape[1] != 2 or arcs.shape[1] != 2:
        raise ValueError(f"Unexpected column count in trip {trip_id}")
    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError(f"Arc references an invalid node in trip {trip_id}")
    if np.any(route < 0) or np.any(route >= len(arcs)):
        raise ValueError(f"Route references an invalid arc in trip {trip_id}")
    return track, nodes, arcs, route


def bearing_deg(dx, dy):
    # Bearings measured clockwise from north.
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_difference(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def project_trip(track, nodes):
    lon0 = float(np.nanmedian(track[:, 0]))
    lat0 = float(np.nanmedian(track[:, 1]))
    zone = int(np.floor((lon0 + 180) / 6) + 1)
    epsg = (32600 if lat0 >= 0 else 32700) + zone
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    tx, ty = transformer.transform(track[:, 0], track[:, 1])
    nx, ny = transformer.transform(nodes[:, 0], nodes[:, 1])
    return np.column_stack([tx, ty]), np.column_stack([nx, ny]), epsg


def build_trip_examples(folder: Path, k_candidates=20):
    track, nodes_ll, arcs, route = read_trip(folder)
    gps, nodes, epsg = project_trip(track, nodes_ll)
    starts, ends = nodes[arcs[:, 0]], nodes[arcs[:, 1]]
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    midpoints = (starts + ends) / 2.0
    # A midpoint KD-tree provides a fast shortlist; exact point-to-segment
    # distances are then calculated for each shortlisted candidate.
    tree = cKDTree(midpoints)
    k = min(max(1, int(k_candidates)), len(arcs))
    route_unique = np.unique(route)
    route_starts, route_ends = starts[route_unique], ends[route_unique]
    route_vec = route_ends - route_starts
    route_len2 = np.sum(route_vec * route_vec, axis=1)
    route_len2[route_len2 == 0] = 1e-12

    # GPS heading/speed from adjacent observations; endpoints use one-sided deltas.
    n = len(gps)
    delta = np.zeros_like(gps)
    if n > 1:
        delta[1:-1] = gps[2:] - gps[:-2]
        delta[0] = gps[1] - gps[0]
        delta[-1] = gps[-1] - gps[-2]
    heading = bearing_deg(delta[:, 0], delta[:, 1])
    ts = track[:, 2]
    # Match the centered displacement to its actual elapsed time.
    elapsed = np.ones(n, dtype=float)
    if n > 1:
        elapsed[0] = max(float(ts[1] - ts[0]), 1.0)
        elapsed[-1] = max(float(ts[-1] - ts[-2]), 1.0)
    if n > 2:
        elapsed[1:-1] = np.maximum(ts[2:] - ts[:-2], 1.0)
    speed = np.linalg.norm(delta, axis=1) / elapsed
    dt = elapsed

    rows, labels, groups, point_ids, candidate_arc_ids = [], [], [], [], []
    candidate_misses = 0
    for i, point in enumerate(gps):
        # Build a geometric proxy label: closest arc among this trip's reference route.
        relr = point - route_starts
        fracr = np.clip(np.sum(relr * route_vec, axis=1) / route_len2, 0.0, 1.0)
        projr = route_starts + fracr[:, None] * route_vec
        route_dist = np.linalg.norm(projr - point, axis=1)
        true_arc = int(route_unique[int(np.argmin(route_dist))])

        _, cand = tree.query(point, k=k)
        cand = np.atleast_1d(cand).astype(int)
        rel = point - starts[cand]
        lensq = np.sum(vectors[cand] * vectors[cand], axis=1)
        lensq = np.where(lensq == 0, 1e-12, lensq)
        frac = np.clip(np.sum(rel * vectors[cand], axis=1) / lensq, 0.0, 1.0)
        projected = starts[cand] + frac[:, None] * vectors[cand]
        dist = np.linalg.norm(projected - point, axis=1)
        dstart = np.linalg.norm(starts[cand] - point, axis=1)
        dend = np.linalg.norm(ends[cand] - point, axis=1)
        arc_heading = bearing_deg(vectors[cand, 0], vectors[cand, 1])
        hdiff = angle_difference(heading[i], arc_heading)
        order = np.argsort(dist)
        rank = np.empty(len(cand), dtype=float)
        rank[order] = np.arange(1, len(cand) + 1, dtype=float)

        if true_arc not in cand:
            candidate_misses += 1
        for j, arc_id in enumerate(cand):
            rows.append([dist[j], dstart[j], dend[j], lengths[arc_id],
                         hdiff[j], speed[i], dt[i], frac[j], rank[j]])
            labels.append(int(arc_id == true_arc))
            groups.append(folder.name)
            point_ids.append(i)
            candidate_arc_ids.append(int(arc_id))

    frame = pd.DataFrame(rows, columns=FEATURES)
    frame["label"] = labels
    frame["trip_id"] = groups
    frame["gps_index"] = point_ids
    frame["arc_id"] = candidate_arc_ids
    stats = {
        "trip_id": folder.name, "gps_observations": int(len(track)),
        "nodes": int(len(nodes)), "arcs": int(len(arcs)),
        "reference_route_arcs": int(len(route)), "candidate_k": int(k),
        "candidate_misses": int(candidate_misses),
        "candidate_miss_rate": float(candidate_misses / max(1, len(track))),
        "projection_epsg": int(epsg)
    }
    return frame, stats


def build_partition(partition: Path, k_candidates):
    frames, stats = [], []
    trip_dirs = sorted(p for p in partition.iterdir() if p.is_dir())
    if not trip_dirs:
        raise RuntimeError(f"No trip folders found in {partition}")
    for idx, folder in enumerate(trip_dirs, 1):
        print(f"[{idx}/{len(trip_dirs)}] Processing trip {folder.name} ...", flush=True)
        df, st = build_trip_examples(folder, k_candidates)
        frames.append(df)
        stats.append(st)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(stats)


def evaluate_pointwise(frame, probabilities):
    temp = frame[["trip_id", "gps_index", "candidate_rank", "label"]].copy()
    temp["score"] = probabilities
    best = (
        temp.sort_values(
            ["trip_id", "gps_index", "score", "candidate_rank"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .drop_duplicates(["trip_id", "gps_index"], keep="first")
    )
    coverage = frame.groupby(["trip_id", "gps_index"])["label"].sum()
    covered_keys = coverage[coverage > 0].index
    best_index = pd.MultiIndex.from_frame(best[["trip_id", "gps_index"]])
    best_covered = best.loc[best_index.isin(covered_keys)]
    observation_count = int(len(best))
    covered_count = int(len(best_covered))
    return {
        "candidate_observations": observation_count,
        "candidate_covered_observations": covered_count,
        "candidate_missed_observations": observation_count - covered_count,
        "candidate_coverage_rate": float(covered_count / max(1, observation_count)),
        "top1_proxy_accuracy_on_candidate_covered_points": (
            float(best_covered["label"].mean()) if covered_count else None
        ),
        "top1_proxy_accuracy_on_all_observations": (
            float(best["label"].mean()) if observation_count else None
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Student folder", help="Folder containing train/ and student_test/")
    ap.add_argument("--k", type=int, default=20, help="Number of nearest road-arc candidates per GPS point")
    ap.add_argument("--output", default="artifacts")
    ap.add_argument("--search-iterations", type=int, default=24)
    ap.add_argument("--cv-folds", type=int, default=4)
    ap.add_argument("--cv-points-per-trip", type=int, default=250)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument(
        "--reuse-features",
        action="store_true",
        help="Reuse matching-k candidate CSVs from the output directory.",
    )
    args = ap.parse_args()
    if args.search_iterations < 1 or args.cv_folds < 2 or args.cv_points_per_trip < 1:
        ap.error("Search iterations and CV points must be positive; CV folds must be at least 2.")
    root = Path(args.data)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    start = time.time()
    old_metrics_path = out / "decision_tree_metrics.json"
    historical_metrics = (
        json.loads(old_metrics_path.read_text(encoding="utf-8"))
        if old_metrics_path.exists()
        else None
    )
    if historical_metrics and "cv_design" in historical_metrics:
        historical_previous = historical_metrics.get("historical_previous_run")
        historical_note = historical_metrics.get("historical_previous_run_note")
    elif historical_metrics:
        historical_previous = historical_metrics.get("student_development")
        historical_note = (
            "The prior metric named top1_accuracy_on_candidate_covered_points "
            "actually included all observations, including candidate misses. "
            "The corrected metrics above explicitly report covered-only and "
            "all-observation results."
        )
    else:
        historical_previous = None
        historical_note = None
    feature_paths = {
        "train": out / "train_candidate_features.csv",
        "student": out / "student_candidate_features.csv",
        "train_stats": out / "train_trip_diagnostics.csv",
        "student_stats": out / "student_trip_diagnostics.csv",
    }
    if args.reuse_features:
        missing_cache = [str(path) for path in feature_paths.values() if not path.exists()]
        bundle_path = out / "decision_tree_model.joblib"
        if missing_cache or not bundle_path.exists():
            raise FileNotFoundError(
                f"Cannot reuse features; missing cached inputs or model bundle: "
                f"{missing_cache or [str(bundle_path)]}"
            )
        cached_bundle = joblib.load(bundle_path)
        if cached_bundle.get("candidate_k") != args.k:
            raise ValueError(
                f"Cached feature k does not match --k {args.k}; "
                "rebuild candidate features instead."
            )
        source_partitions = [root / "train", root / "student_test"]
        if any(not partition.is_dir() for partition in source_partitions):
            raise FileNotFoundError("Expected train/ and student_test/ data partitions.")
        cache_time = min(
            feature_paths["train"].stat().st_mtime,
            feature_paths["student"].stat().st_mtime,
        )
        newest_source = max(
            path.stat().st_mtime
            for partition in source_partitions
            for path in partition.rglob("*")
            if path.is_file()
        )
        if newest_source > cache_time:
            raise ValueError(
                "Dataset files are newer than cached features; rerun without "
                "--reuse-features."
            )
        train = pd.read_csv(feature_paths["train"], dtype={"trip_id": str})
        dev = pd.read_csv(feature_paths["student"], dtype={"trip_id": str})
        train_stats = pd.read_csv(feature_paths["train_stats"])
        dev_stats = pd.read_csv(feature_paths["student_stats"])
        for frame_name, frame, partition in (
            ("training", train, source_partitions[0]),
            ("student development", dev, source_partitions[1]),
        ):
            expected_trips = {path.name for path in partition.iterdir() if path.is_dir()}
            if set(frame["trip_id"].astype(str).unique()) != expected_trips:
                raise ValueError(
                    f"Cached {frame_name} trip IDs do not match the current dataset."
                )
    else:
        train, train_stats = build_partition(root / "train", args.k)
        dev, dev_stats = build_partition(root / "student_test", args.k)
        train.to_csv(feature_paths["train"], index=False)
        dev.to_csv(feature_paths["student"], index=False)
        train_stats.to_csv(feature_paths["train_stats"], index=False)
        dev_stats.to_csv(feature_paths["student_stats"], index=False)

    # Keep the existing proxy-label rule; uncovered candidate sets cannot train ranking.
    covered_keys = train.groupby(["trip_id", "gps_index"]).label.transform("sum") > 0
    train_fit = train.loc[covered_keys].reset_index(drop=True)
    if train_fit.empty or train_fit["label"].nunique() != 2:
        raise RuntimeError("Training data must contain both proxy-label classes on covered points.")

    # CV uses a reproducible point sample; each sampled GPS point keeps all candidates.
    cv_sample = sample_cv_points(train_fit, args.cv_points_per_trip, random_state=42)
    X_cv = cv_sample.loc[:, FEATURES + ["trip_id", "gps_index"]]
    y_cv = cv_sample["label"].to_numpy(dtype=int)
    groups = cv_sample["trip_id"].to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=args.cv_folds, shuffle=True, random_state=42
    )
    cv_splits = list(splitter.split(X_cv, y_cv, groups=groups))
    scoring = {
        "top1_accuracy": top1_proxy_accuracy_scorer,
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }
    metric_names = list(scoring)

    # Refit the original settings on the same full training data for a fair comparison.
    baseline_cv = cross_validate(
        SearchDecisionTree(**ORIGINAL_PARAMS),
        X_cv,
        y_cv,
        cv=cv_splits,
        scoring=scoring,
        n_jobs=args.n_jobs,
        return_train_score=False,
    )
    baseline_cv_summary = summarize_cv_results(baseline_cv, metric_names)

    param_distributions = {
        "criterion": ["gini", "entropy"],
        "max_depth": [8, 12, 16, 24, None],
        "min_samples_split": [20, 50, 100],
        "min_samples_leaf": [5, 10, 20],
        "max_features": [None, "sqrt", 0.8],
        "class_weight": [None, "balanced", {0: 1, 1: 2}],
        "ccp_alpha": [0.0, 1e-5, 1e-4, 1e-3],
    }
    search = RandomizedSearchCV(
        SearchDecisionTree(random_state=42),
        param_distributions=param_distributions,
        n_iter=args.search_iterations,
        scoring=scoring,
        refit="top1_accuracy",
        cv=cv_splits,
        random_state=42,
        n_jobs=args.n_jobs,
        verbose=1,
        return_train_score=False,
    )
    search_start = time.time()
    search.fit(X_cv, y_cv)
    search_seconds = time.time() - search_start
    best_params = dict(search.best_params_)
    best_params["random_state"] = 42
    improved_cv_summary = summarize_search_cv(search, metric_names)

    # Fit both selected and original settings on all covered training candidate rows.
    baseline_model = DecisionTreeClassifier(**ORIGINAL_PARAMS)
    baseline_start = time.time()
    baseline_model.fit(train_fit.loc[:, FEATURES], train_fit["label"])
    baseline_fit_seconds = time.time() - baseline_start
    model = DecisionTreeClassifier(**best_params)
    improved_start = time.time()
    model.fit(train_fit.loc[:, FEATURES], train_fit["label"])
    improved_fit_seconds = time.time() - improved_start

    baseline_prob = positive_class_probabilities(baseline_model, dev)
    improved_prob = positive_class_probabilities(model, dev)
    baseline_dev = {
        "candidate_rows": evaluate_candidate_rows(dev, baseline_prob),
        "pointwise_proxy": evaluate_pointwise(dev, baseline_prob),
        "fit_seconds": baseline_fit_seconds,
    }
    improved_dev = {
        "candidate_rows": evaluate_candidate_rows(dev, improved_prob),
        "pointwise_proxy": evaluate_pointwise(dev, improved_prob),
        "fit_seconds": improved_fit_seconds,
    }
    original_bundle = {
        "model": baseline_model,
        "features": FEATURES.copy(),
        "feature_order": FEATURES.copy(),
        "candidate_k": args.k,
        "best_hyperparameters": ORIGINAL_PARAMS.copy(),
        "label_method": (
            "nearest reference-route arc to each GPS observation; geometric proxy"
        ),
    }
    joblib.dump(original_bundle, out / "decision_tree_model_original.joblib")
    score_bundle = {
        "model": model,
        "features": FEATURES.copy(),
        "feature_order": FEATURES.copy(),
        "candidate_k": args.k,
        "best_hyperparameters": best_params,
        "label_method": (
            "nearest reference-route arc to each GPS observation; geometric proxy"
        ),
        "prediction_output": (
            "P(proxy-positive) per candidate row; preserve trip_id, gps_index, and arc_id"
        ),
    }
    joblib.dump(score_bundle, out / "decision_tree_model.joblib")
    score_candidates(score_bundle, dev).to_csv(
        out / "student_candidate_scores.csv", index=False
    )

    cv_results = pd.DataFrame(
        {
            "rank_test_top1_accuracy": search.cv_results_["rank_test_top1_accuracy"],
            "mean_test_top1_accuracy": search.cv_results_["mean_test_top1_accuracy"],
            "std_test_top1_accuracy": search.cv_results_["std_test_top1_accuracy"],
            "mean_test_accuracy": search.cv_results_["mean_test_accuracy"],
            "std_test_accuracy": search.cv_results_["std_test_accuracy"],
            "mean_test_precision": search.cv_results_["mean_test_precision"],
            "std_test_precision": search.cv_results_["std_test_precision"],
            "mean_test_recall": search.cv_results_["mean_test_recall"],
            "std_test_recall": search.cv_results_["std_test_recall"],
            "mean_test_f1": search.cv_results_["mean_test_f1"],
            "std_test_f1": search.cv_results_["std_test_f1"],
            "params": [
                json.dumps(params, sort_keys=True, default=str)
                for params in search.cv_results_["params"]
            ],
        }
    )
    cv_results.to_csv(out / "decision_tree_cv_results.csv", index=False)

    report_depth = min(4, int(model.get_depth()))
    (out / "decision_tree_rules.txt").write_text(
        export_text(model, feature_names=FEATURES, max_depth=report_depth),
        encoding="utf-8",
    )
    fig, ax = plt.subplots(figsize=(24, 14))
    plot_tree(
        model,
        feature_names=FEATURES,
        class_names=[str(value) for value in model.classes_],
        max_depth=report_depth,
        filled=True,
        rounded=True,
        impurity=False,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(out / "decision_tree_visualization.png", dpi=160)
    fig.savefig(out / "decision_tree_visualization.svg")
    plt.close(fig)

    comparisons = []
    for name, result in (
        ("Original settings, full training data", baseline_dev),
        ("Tuned settings, full training data", improved_dev),
    ):
        comparisons.append(
            {
                "model": name,
                "accuracy": result["candidate_rows"]["accuracy"],
                "precision": result["candidate_rows"]["precision"],
                "recall": result["candidate_rows"]["recall"],
                "f1": result["candidate_rows"]["f1"],
                "top1_proxy_accuracy_on_covered_points": result["pointwise_proxy"][
                    "top1_proxy_accuracy_on_candidate_covered_points"
                ],
                "top1_proxy_accuracy_on_all_points": result["pointwise_proxy"][
                    "top1_proxy_accuracy_on_all_observations"
                ],
            }
        )
    report = {
        "model": "DecisionTreeClassifier",
        "features": FEATURES,
        "feature_order": FEATURES,
        "training_candidate_rows_all": int(len(train)),
        "training_candidate_rows_used": int(len(train_fit)),
        "student_candidate_rows": int(len(dev)),
        "training_trips": int(train.trip_id.nunique()),
        "student_trips": int(dev.trip_id.nunique()),
        "train_candidate_miss_rate_mean_by_trip": float(
            train_stats.candidate_miss_rate.mean()
        ),
        "student_candidate_miss_rate_mean_by_trip": float(
            dev_stats.candidate_miss_rate.mean()
        ),
        "label_method": (
            "For each GPS observation, select the geometrically nearest arc among "
            "that trip's unique reference-route arcs; candidate arc equality is the "
            "binary proxy label."
        ),
        "cv_design": {
            "search": "RandomizedSearchCV",
            "folds": args.cv_folds,
            "splitter": "StratifiedGroupKFold",
            "grouping": "trip_id",
            "sampled_gps_points_per_trip_max": args.cv_points_per_trip,
            "sampled_candidate_rows": int(len(cv_sample)),
            "search_iterations": args.search_iterations,
            "selection_metric": "top1_accuracy on covered GPS points (proxy labels)",
            "tie_break": "lowest candidate_rank (smallest exact point-to-segment distance)",
            "original_settings": ORIGINAL_PARAMS,
            "original_cross_validation": baseline_cv_summary,
            "best_cross_validation": improved_cv_summary,
            "best_hyperparameters": best_params,
            "search_seconds": search_seconds,
        },
        "student_development": {
            "original_settings_refit_on_all_training_data": baseline_dev,
            "tuned_settings": improved_dev,
            "comparison": comparisons,
        },
        "historical_previous_run": historical_previous,
        "historical_previous_run_note": historical_note,
        "warning": (
            "All labels are geometrically derived proxy labels, not GPS-row ground "
            "truth. CV and development scores must not be interpreted as definitive "
            "route-matching accuracy. Student development data was not used for "
            "hyperparameter selection."
        ),
        "total_seconds": time.time() - start,
    }
    (out / "decision_tree_metrics.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print("\n=== DECISION TREE RESULTS ===")
    print(json.dumps(report, indent=2))
    print(f"\nSaved model: {out / 'decision_tree_model.joblib'}")

if __name__ == "__main__":
    main()
