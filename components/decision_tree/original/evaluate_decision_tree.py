"""Evaluate the saved PathSnap Decision Tree at GPS-observation level."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from decision_tree import FEATURES, positive_class_probabilities


IDENTIFIERS = ["trip_id", "gps_index", "arc_id", "candidate_rank", "label"]
CV_SEED = 42


def load_training_cv_sample(csv_path: Path, points_per_trip: int) -> pd.DataFrame:
    """Read only a deterministic sample of complete covered GPS candidate sets."""
    point_labels: dict[str, set[int]] = {}
    for chunk in pd.read_csv(
        csv_path,
        usecols=["trip_id", "gps_index", "label"],
        dtype={"trip_id": str},
        chunksize=250_000,
    ):
        positives = chunk.loc[chunk["label"].eq(1), ["trip_id", "gps_index"]]
        for trip_id, trip_points in positives.groupby("trip_id", sort=False):
            point_labels.setdefault(trip_id, set()).update(
                trip_points["gps_index"].astype(int).tolist()
            )

    rng = np.random.default_rng(CV_SEED)
    selected: dict[str, set[int]] = {}
    for trip_id, point_ids in sorted(point_labels.items()):
        ordered_ids = np.asarray(sorted(point_ids), dtype=np.int64)
        if len(ordered_ids) > points_per_trip:
            ordered_ids = rng.choice(ordered_ids, size=points_per_trip, replace=False)
        selected[trip_id] = set(ordered_ids.tolist())

    columns = FEATURES + IDENTIFIERS
    sampled_frames = []
    for chunk in pd.read_csv(
        csv_path,
        usecols=columns,
        dtype={"trip_id": str},
        chunksize=250_000,
    ):
        for trip_id, trip_rows in chunk.groupby("trip_id", sort=False):
            selected_points = selected.get(trip_id)
            if selected_points:
                sampled_frames.append(
                    trip_rows.loc[trip_rows["gps_index"].isin(selected_points)]
                )
    if not sampled_frames:
        raise ValueError(f"No covered training rows found in {csv_path}")
    sample = pd.concat(sampled_frames, ignore_index=True)
    if sample.groupby(["trip_id", "gps_index"])["label"].sum().eq(0).any():
        raise ValueError("CV sample contains an observation without a positive candidate.")
    return sample


def observation_metrics(
    frame: pd.DataFrame, probabilities: np.ndarray
) -> dict[str, float | int | None]:
    """Evaluate one selected candidate per GPS observation.

    Macro F1 uses trip-qualified arc IDs as classes and only covered points.
    Log loss is the multiclass NLL after normalizing positive-class candidate
    probabilities within each GPS point; it also excludes shortlist misses.
    """
    if len(frame) != len(probabilities):
        raise ValueError("Candidate rows and probabilities must have equal length.")
    scored = frame[["trip_id", "gps_index", "arc_id", "candidate_rank", "label"]].copy()
    scored["probability"] = np.asarray(probabilities, dtype=float)
    scored["arc_class"] = (
        scored["trip_id"].astype(str) + ":" + scored["arc_id"].astype(str)
    )
    scored["true_arc_class"] = scored["arc_class"].where(scored["label"].eq(1))

    groups = ["trip_id", "gps_index"]
    ranked = scored.sort_values(
        groups + ["probability", "candidate_rank"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    best = ranked.drop_duplicates(groups, keep="first").copy()
    point_coverage = scored.groupby(groups, sort=False)["label"].sum().gt(0)
    best_index = pd.MultiIndex.from_frame(best[groups])
    covered_best = best.loc[best_index.isin(point_coverage[point_coverage].index)].copy()

    if covered_best.empty:
        macro_f1 = None
    else:
        true_arcs = (
            scored.loc[scored["label"].eq(1), groups + ["true_arc_class"]]
            .drop_duplicates(groups)
            .rename(columns={"true_arc_class": "actual_arc_class"})
        )
        covered_best = covered_best.merge(true_arcs, on=groups, how="inner")
        macro_f1 = float(
            f1_score(
                covered_best["actual_arc_class"],
                covered_best["arc_class"],
                average="macro",
                zero_division=0,
            )
        )

    totals = scored.groupby(groups, sort=False)["probability"].sum()
    true_probability = (
        scored.loc[scored["label"].eq(1)]
        .groupby(groups, sort=False)["probability"]
        .sum()
    )
    covered_keys = point_coverage[point_coverage].index
    covered_totals = totals.reindex(covered_keys)
    valid_totals = covered_totals.loc[covered_totals.gt(0)]
    valid_true = true_probability.reindex(valid_totals.index)
    if len(valid_totals):
        normalized_true_probability = (valid_true / valid_totals).clip(1e-15, 1.0)
        log_loss = float(-np.log(normalized_true_probability).mean())
    else:
        log_loss = None

    observations = int(len(best))
    covered_count = int(point_coverage.sum())
    correct_count = int(best["label"].sum())
    return {
        "observations": observations,
        "candidate_covered_observations": covered_count,
        "candidate_missed_observations": observations - covered_count,
        "candidate_coverage_rate": float(covered_count / max(1, observations)),
        "top1_correct_observations": correct_count,
        "top1_accuracy_all_observations": float(correct_count / max(1, observations)),
        "top1_accuracy_covered_observations": (
            float(covered_best["label"].mean()) if covered_count else None
        ),
        "macro_f1_arc_class_covered_observations": macro_f1,
        "normalized_candidate_log_loss_covered_observations": log_loss,
        "log_loss_observations": int(len(valid_totals)),
        "log_loss_zero_probability_observations": int(
            covered_count - len(valid_totals)
        ),
    }


def mean_metric(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row[key] is not None]
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/Student folder")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--cv-points-per-trip", type=int, default=250)
    args = parser.parse_args()
    if args.cv_folds < 2 or args.cv_points_per_trip < 1:
        parser.error("--cv-folds must be at least 2 and --cv-points-per-trip positive.")

    data_root = Path(args.data)
    output = Path(args.output)
    train_csv = output / "train_candidate_features.csv"
    student_csv = output / "student_candidate_features.csv"
    model_path = output / "decision_tree_model.joblib"
    metrics_path = output / "decision_tree_metrics.json"
    required = [data_root / "train", data_root / "student_test"]
    missing = [str(path) for path in [*required, train_csv, student_csv, model_path, metrics_path]
               if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required training/development inputs are missing: {missing}")
    if not required[0].is_dir() or not required[1].is_dir():
        raise NotADirectoryError("Expected data/train and data/student_test partitions.")

    bundle = joblib.load(model_path)
    if bundle.get("feature_order") != FEATURES or bundle.get("candidate_k") != 20:
        raise ValueError("Saved model feature order or candidate k does not match this project.")
    model = bundle["model"]
    best_params = model.get_params()

    train_sample = load_training_cv_sample(train_csv, args.cv_points_per_trip)
    X = train_sample.loc[:, FEATURES]
    y = train_sample["label"].to_numpy(dtype=int)
    groups = train_sample["trip_id"].to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=args.cv_folds, shuffle=True, random_state=CV_SEED
    )
    splits = list(splitter.split(X, y, groups=groups))

    fold_train_metrics = []
    fold_validation_metrics = []
    validation_frames = []
    validation_probabilities = []
    for fold_number, (train_idx, validation_idx) in enumerate(splits, start=1):
        fold_model = type(model)(**best_params)
        fold_train = train_sample.iloc[train_idx].reset_index(drop=True)
        fold_validation = train_sample.iloc[validation_idx].reset_index(drop=True)
        fold_model.fit(fold_train.loc[:, FEATURES], fold_train["label"])
        train_probabilities = positive_class_probabilities(fold_model, fold_train)
        validation_fold_probabilities = positive_class_probabilities(
            fold_model, fold_validation
        )
        train_metrics = observation_metrics(fold_train, train_probabilities)
        validation_metrics = observation_metrics(
            fold_validation, validation_fold_probabilities
        )
        train_metrics["fold"] = fold_number
        validation_metrics["fold"] = fold_number
        fold_train_metrics.append(train_metrics)
        fold_validation_metrics.append(validation_metrics)
        validation_frames.append(fold_validation)
        validation_probabilities.append(validation_fold_probabilities)
        print(
            f"Fold {fold_number}/{args.cv_folds}: "
            f"train top-1={train_metrics['top1_accuracy_covered_observations']:.4f}, "
            f"validation top-1={validation_metrics['top1_accuracy_covered_observations']:.4f}",
            flush=True,
        )

    pooled_validation = pd.concat(validation_frames, ignore_index=True)
    pooled_validation_probabilities = np.concatenate(validation_probabilities)
    validation_oof_metrics = observation_metrics(
        pooled_validation, pooled_validation_probabilities
    )
    train_sample_probabilities = positive_class_probabilities(model, train_sample)
    train_in_sample_metrics = observation_metrics(
        train_sample, train_sample_probabilities
    )

    student = pd.read_csv(student_csv, dtype={"trip_id": str})
    student_probabilities = positive_class_probabilities(model, student)
    student_metrics = observation_metrics(student, student_probabilities)

    cv_train_summary = {
        key: mean_metric(fold_train_metrics, key)
        for key in (
            "top1_accuracy_covered_observations",
            "macro_f1_arc_class_covered_observations",
            "normalized_candidate_log_loss_covered_observations",
        )
    }
    cv_validation_summary = {
        key: validation_oof_metrics[key]
        for key in (
            "top1_accuracy_covered_observations",
            "top1_accuracy_all_observations",
            "macro_f1_arc_class_covered_observations",
            "normalized_candidate_log_loss_covered_observations",
        )
    }
    train_validation_gap = {
        "top1_accuracy_covered_observations": (
            cv_train_summary["top1_accuracy_covered_observations"]
            - cv_validation_summary["top1_accuracy_covered_observations"]
        ),
        "macro_f1_arc_class_covered_observations": (
            cv_train_summary["macro_f1_arc_class_covered_observations"]
            - cv_validation_summary["macro_f1_arc_class_covered_observations"]
        ),
        "normalized_candidate_log_loss_covered_observations": (
            cv_train_summary["normalized_candidate_log_loss_covered_observations"]
            - cv_validation_summary["normalized_candidate_log_loss_covered_observations"]
        ),
    }

    results = []
    for row in fold_train_metrics:
        results.append({"split": "training_fold", **row})
    for row in fold_validation_metrics:
        results.append({"split": "validation_fold", **row})
    results.extend(
        [
            {"split": "training_cv_mean", **cv_train_summary},
            {"split": "validation_oof_pooled", **validation_oof_metrics},
            {"split": "training_sample_in_sample", **train_in_sample_metrics},
            {"split": "student_test", **student_metrics},
        ]
    )
    results_csv = output / "decision_tree_results.csv"
    pd.DataFrame(results).to_csv(results_csv, index=False)

    macro_gap = train_validation_gap["macro_f1_arc_class_covered_observations"]
    top1_gap = train_validation_gap["top1_accuracy_covered_observations"]
    training_log_loss = train_validation_gap[
        "normalized_candidate_log_loss_covered_observations"
    ]
    train_macro = cv_train_summary["macro_f1_arc_class_covered_observations"]
    val_macro = cv_validation_summary["macro_f1_arc_class_covered_observations"]
    if top1_gap is not None and macro_gap is not None and (top1_gap >= 0.05 or macro_gap >= 0.05):
        fit_conclusion = (
            "Evidence of a training-to-validation generalization gap; possible "
            "overfitting. Interpret cautiously because the same grouped CV folds "
            "were used for hyperparameter selection."
        )
    elif training_log_loss is not None and -training_log_loss >= 0.1:
        fit_conclusion = (
            "Top-1 and macro-F1 show only a small train/validation gap, so there "
            "is no clear ranking underfit/overfit signal. Validation normalized "
            "log loss is materially higher than training log loss, indicating "
            "less reliable candidate probabilities on held-out trips (possible "
            "probability overfit or trip-distribution shift). The depth-limited "
            "tree is not demonstrably too complex or too simple from these "
            "proxy metrics alone."
        )
    elif train_macro is not None and val_macro is not None and train_macro < 0.5 and val_macro < 0.5:
        fit_conclusion = (
            "Both training and validation arc-class macro F1 are low; possible "
            "underfitting, although this many-class proxy task and label noise "
            "make a simple threshold inconclusive."
        )
    else:
        fit_conclusion = (
            "No large training/validation gap is evident under the reported "
            "observation-level proxy metrics. This does not establish that the "
            "model is well calibrated or that route matching is accurate."
        )

    evaluation = {
        "evaluation_source": {
            "training_partition": str(required[0]),
            "student_testing_partition": str(required[1]),
            "training_trips": int(train_sample["trip_id"].nunique()),
            "student_testing_trips": int(student["trip_id"].nunique()),
            "training_cv_sample_candidate_rows": int(len(train_sample)),
            "training_cv_sample_observations": int(
                train_sample[["trip_id", "gps_index"]].drop_duplicates().shape[0]
            ),
            "student_testing_candidate_rows": int(len(student)),
            "student_testing_observations": int(
                student[["trip_id", "gps_index"]].drop_duplicates().shape[0]
            ),
            "reserved_evaluator_data_accessed": False,
        },
        "evaluation_method": {
            "prediction_unit": "one highest-probability candidate per GPS observation",
            "tie_break": "lowest candidate_rank",
            "label_method": bundle["label_method"],
            "student_top1_all_points_counts_candidate_misses_as_incorrect": True,
            "macro_f1": (
                "Macro F1 over trip-qualified arc IDs, among candidate-covered "
                "observations; one selected arc prediction per observation."
            ),
            "log_loss": (
                "Multiclass negative log likelihood among candidate-covered "
                "observations after normalizing per-candidate positive-class "
                "probabilities to sum to one within each observation."
            ),
            "validation": (
                f"{args.cv_folds}-fold StratifiedGroupKFold on training trips only; "
                "all candidates for a sampled GPS observation remain together."
            ),
            "validation_caveat": (
                "These are the same training-only folds used for hyperparameter "
                "selection, so validation summaries are not an untouched final test."
            ),
        },
        "model": {
            "class": type(model).__name__,
            "parameters": best_params,
            "feature_order": bundle["feature_order"],
            "actual_tree_depth": int(model.get_depth()),
            "leaf_count": int(model.get_n_leaves()),
            "node_count": int(model.tree_.node_count),
        },
        "training_cv": {
            "fold_training_macro_metrics_mean": cv_train_summary,
            "fold_training_macro_metrics_by_fold": fold_train_metrics,
            "out_of_fold_validation_metrics": validation_oof_metrics,
            "validation_metrics_by_fold": fold_validation_metrics,
            "training_validation_gap": train_validation_gap,
            "full_model_training_sample_in_sample": train_in_sample_metrics,
            "conclusion": fit_conclusion,
        },
        "student_testing": student_metrics,
        "artifacts": {
            "results_csv": str(results_csv),
            "metrics_json": str(metrics_path),
            "report_txt": str(output / "decision_tree_evaluation_report.txt"),
        },
        "proxy_label_warning": (
            "These metrics measure agreement with geometric proxy labels, not "
            "definitive GPS-row route-matching ground truth. The .route files "
            "do not align route entries to GPS observations."
        ),
    }

    previous_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    previous_metrics["first_evaluation"] = evaluation
    metrics_path.write_text(
        json.dumps(previous_metrics, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    report_lines = [
        "PathSnap Decision Tree — First Evaluation Results",
        "=" * 54,
        "",
        "Dataset used",
        f"- Training: {evaluation['evaluation_source']['training_trips']} trips "
        f"from {required[0]} (sampled only for CV diagnostics).",
        f"- Student testing/development: {evaluation['evaluation_source']['student_testing_trips']} "
        f"trips from {required[1]}.",
        "- Reserved evaluator-testing data accessed: No.",
        "- Original dataset split and proxy-label generation method changed: No.",
        "",
        "Model configuration",
        f"- Model: {type(model).__name__}",
        f"- Candidate k: {bundle['candidate_k']}",
        f"- Hyperparameters: {json.dumps(best_params, sort_keys=True)}",
        f"- Fitted tree structure: depth {model.get_depth()}, "
        f"{model.get_n_leaves()} leaves, {model.tree_.node_count} nodes.",
        f"- Feature order: {', '.join(bundle['feature_order'])}",
        "",
        "Student-testing observation-level results",
        f"- Observations evaluated: {student_metrics['observations']}",
        f"- Candidate-covered observations: {student_metrics['candidate_covered_observations']}",
        f"- Candidate misses: {student_metrics['candidate_missed_observations']}",
        f"- Top-1 accuracy, all observations: "
        f"{student_metrics['top1_accuracy_all_observations']:.6f}",
        f"- Top-1 accuracy, covered observations: "
        f"{student_metrics['top1_accuracy_covered_observations']:.6f}",
        f"- Macro F1, covered observations, trip-qualified arc classes: "
        f"{student_metrics['macro_f1_arc_class_covered_observations']:.6f}",
        f"- Normalized candidate log loss, covered observations: "
        f"{student_metrics['normalized_candidate_log_loss_covered_observations']:.6f}",
        f"- Observations included in log loss: {student_metrics['log_loss_observations']}",
        "",
        "Training versus validation",
        "- Cross-validation: "
        f"{args.cv_folds}-fold StratifiedGroupKFold grouped by trip ID; "
        f"{evaluation['evaluation_source']['training_cv_sample_observations']} "
        "covered GPS observations sampled from training data.",
        f"- Mean fold-training top-1 accuracy (covered): "
        f"{cv_train_summary['top1_accuracy_covered_observations']:.6f}",
        f"- Pooled out-of-fold validation top-1 accuracy (covered): "
        f"{validation_oof_metrics['top1_accuracy_covered_observations']:.6f}",
        f"- Mean fold-training macro F1: "
        f"{cv_train_summary['macro_f1_arc_class_covered_observations']:.6f}",
        f"- Pooled out-of-fold validation macro F1: "
        f"{validation_oof_metrics['macro_f1_arc_class_covered_observations']:.6f}",
        f"- Mean fold-training normalized log loss: "
        f"{cv_train_summary['normalized_candidate_log_loss_covered_observations']:.6f}",
        f"- Pooled out-of-fold validation normalized log loss: "
        f"{validation_oof_metrics['normalized_candidate_log_loss_covered_observations']:.6f}",
        f"- Full-model in-sample sampled training top-1: "
        f"{train_in_sample_metrics['top1_accuracy_covered_observations']:.6f}",
        f"- Full-model in-sample sampled training macro F1: "
        f"{train_in_sample_metrics['macro_f1_arc_class_covered_observations']:.6f}",
        f"- Full-model in-sample sampled training normalized log loss: "
        f"{train_in_sample_metrics['normalized_candidate_log_loss_covered_observations']:.6f}",
        f"- Top-1 training minus validation gap: "
        f"{train_validation_gap['top1_accuracy_covered_observations']:.6f}",
        f"- Validation minus training normalized log-loss gap: "
        f"{-train_validation_gap['normalized_candidate_log_loss_covered_observations']:.6f}",
        "",
        "Overfitting/underfitting assessment",
        fit_conclusion,
        "The validation folds were also used to select hyperparameters, so the "
        "validation results may be optimistic. The student-testing scores are "
        "reported after selection and were not used for tuning.",
        "",
        "Metric definitions and limitations",
        evaluation["evaluation_method"]["macro_f1"],
        evaluation["evaluation_method"]["log_loss"],
        evaluation["proxy_label_warning"],
    ]
    report_path = output / "decision_tree_evaluation_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))
    print(f"\nWrote {results_csv}, {metrics_path}, and {report_path}")


if __name__ == "__main__":
    main()
