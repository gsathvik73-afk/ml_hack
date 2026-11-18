# Shared model integration contract

This guide covers DT, incoming logistic regression and KNN, and the later stacking merge. Active model code stays in notebooks. The user assigned the frontend to Claude; this workflow owns model integration, evaluation, exports and backend/API work.

## Read these sources of truth

| Purpose | File |
|---|---|
| Current model paths and integration status | [models/registry.json](../models/registry.json) |
| Active ordered 32 input fields | [results/direction_features/feature_columns.json](../results/direction_features/feature_columns.json) |
| Current training and student-test feature tables | [training Parquet](../outputs/direction_features/train.parquet), [student-test Parquet](../outputs/direction_features/student_test.parquet) |
| Feature rules and configuration | [05_direction_features.ipynb](../05_direction_features.ipynb), [feature config](../outputs/direction_features/config.json) |
| Shared point-to-fold mapping | [outputs/folds/group_folds.csv](../outputs/folds/group_folds.csv) |
| Metric and OOF rules | [outputs/folds/contract.json](../outputs/folds/contract.json) |
| Current exclusions | [outputs/training_exclusions.json](../outputs/training_exclusions.json) |
| Training reference/evaluation metadata | [outputs/candidates/points.csv](../outputs/candidates/points.csv) |
| Student-test reference/evaluation metadata | [outputs/student_test/points.csv](../outputs/student_test/points.csv) |

**Important:** `contract.json` retains the historical 24-feature list. Its fold/metric definitions remain shared, but it is not the active 32-field input schema. Read the feature-list path from the registry. Do not silently append, drop, reorder or zero-fill fields.

The guidelines specify 80 training trips. New runs exclude `00000033` and `00000054` by explicit user instruction, leaving 78. Keep raw files and old all-80 experiments. Preserve the saved fold assignments for the remaining trips and disclose this protocol deviation; do not call it automatically compliant with the original 80-trip requirement.

## The ordered 32 inputs

| Positions | Fields, in order |
|---|---|
| 1-4 | `distance`, `segment_fraction`, `segment_length_m`, `heading_diff_deg` |
| 5-8 | `speed_mps`, `acceleration_mps2`, `time_gap_s`, `distance_rank` |
| 9-12 | `distance_above_nearest_m`, `candidate_count`, `heading_diff_rank`, `distance_prev2_m` |
| 13-16 | `heading_diff_prev2_deg`, `distance_prev_m`, `heading_diff_prev_deg`, `distance_next_m` |
| 17-20 | `heading_diff_next_deg`, `distance_next2_m`, `heading_diff_next2_deg`, `connects_prev` |
| 21-24 | `connects_next`, `start_degree`, `end_degree`, `distance_ratio` |
| 25-28 | `heading_diff_smoothed_3s`, `signed_progress_mps_3s`, `direction_reliability_3s`, `smoothed_speed_mps_3s` |
| 29-32 | `turn_angle_3s`, `heading_diff_smoothed_5s`, `signed_progress_mps_5s`, `direction_reliability_5s` |

Distances and speed calculations use the same per-trip UTM projection as the feature notebooks. Directed heading differences stay between 0 and 180 degrees. Future-point context means these are whole-trip features, not immediate streaming observations. The distance ratio uses a one-metre offset in numerator and denominator; use the saved feature implementation, not a differently scaled replacement.

IDs, fold assignments, `label`, `true_arc_id`, reference route positions, `bad_gps` and reference distances used for target quality are **metadata or targets, never predictors**. The existing `distance` feature is geometric GPS-to-candidate distance, not GPS-to-reference-label distance.

## Labels and candidates must stay shared

- `.route` contains an ordered arc sequence, not one target per GPS row. [Notebook 03](../03_label_points.ipynb) aligns points monotonically to **route positions**, preserving repeated visits to an arc. The assigned arc is a reference-route-derived training label, not independently verified physical truth.
- The alignment cap is 150 m. The training-derived flag threshold is 208.357058 m. Flagged points remain in metadata with `true_arc_id=-1`; their candidate `label` values are missing.
- [Notebook 04](../04_candidates.ipynb) uses exact spatial queries within a frozen 210 m radius and keeps the nearest 100 arcs, with segment ID breaking distance ties. Reversed arcs remain separate candidates.
- `segment_id` is the zero-based `.arcs` row number. `point_id` is the zero-based `.track` row number within its trip. `group_id` identifies that GPS observation across the canonical tables. Preserve trip IDs as strings with leading zeros.
- Drop missing-target candidate rows from supervised fitting and target-based scoring. Do not drop observations from inference merely because reference-based evaluation later flags them.
- A valid point with no positive candidate is an unrecoverable shortlist miss. It remains wrong in Top-1. A point with no candidates also remains in the point-level denominator, even though it has no candidate table row.
- Ranker fitting excludes groups with fewer than two candidates or without exactly one positive. Held-out prediction must still cover valid singleton and no-positive groups. Classifier training exclusions, if any, must be recorded explicitly; they must not silently change the evaluation population.

## Fit preprocessing inside each training split

Read the saved fold mapping; do not make a fresh split. Every trip belongs to one validation fold, and every candidate for a point remains together. Filter the two excluded trips for new runs without modifying their raw data or the historical fold file.

For LR and KNN, provide a fitted pipeline containing the imputer, scaler and classifier. Fit those transformations on the training portion of each fold only. Do not compute medians, scaling parameters or feature selection on validation/student-test rows. A DT may use an imputation pipeline if its fitting procedure needs it; record the actual choice. XGBoost can preserve missing values natively.

After hyperparameters are fixed, fit the final preprocessing pipeline and estimator on all permitted training rows. Keep all candidate rows from a sampled point if runtime sampling is used, and report sample sizes per fold. A sampled or pilot run must not be presented as a full-data final model.

## Incoming model bundle

For a sklearn component, deliver a `model.joblib` **dictionary** with at least these keys. The artifact must be genuinely fitted; a configuration-only object is not a trained model.

| Key | Meaning |
|---|---|
| `model` | Fitted preprocessing-plus-estimator pipeline supporting `predict_proba`. |
| `feature_columns` | Exact ordered list of the 32 raw predictor inputs above. |
| `feature_version` | `direction_32`. |
| `seed` | Recorded seed, normally the shared seed 42. |
| `fold_file` | `outputs/folds/group_folds.csv`. |
| `excluded_trip_ids` | `00000033`, `00000054` for current new runs. |
| `training_trip_ids` | Actual permitted trip IDs used for final fitting. |
| `params` | Actual estimator and preprocessing settings, plus any sampling scope. |

Include the source notebook, dependencies, final artifact, complete OOF candidate scores, complete student-test candidate scores, per-fold metrics, pooled OOF metrics, student-test metrics and fit/predict timings. If preprocessing is delivered separately instead of inside one pipeline, the model manifest must identify every fitted object and its order; do not leave a scaler or imputer behind.

The XGBoost counterpart is its native JSON model plus the same ordered feature-list/configuration metadata. The registry describes how to load each kind. Do not put arbitrary external source code into the runtime loader or execute user component scripts as part of API prediction.

## Canonical candidate-score files

Required OOF columns are **`group_id, segment_id, fold, score`**. `raw_score` is optional. `score` is a normalized probability within a point, not an unnormalized binary classifier probability.

- LR, KNN and DT: take the probability for class 1 and divide by the sum over that point's candidates. If the sum is zero, use a uniform distribution for the point.
- XGBoost ranker: use stable softmax over its raw scores within the point.
- Save every valid held-out candidate, including singleton and candidate-miss groups. A singleton probability is 1. Do not export only each point's winner.
- Sort by `group_id` and deterministic candidate order. Group sizes supplied to the ranker must match those consecutive rows after each split.
- Student-test files use the same candidate keys and normalized `score`; they need not carry a training fold. Predictions must come from the final model trained on permitted training trips only.

### Acceptance checks before registry activation or stacking

1. Check artifact loading and `predict_proba`/native ranker prediction, the positive-class mapping, exact input order and missing-value behaviour.
2. Check IDs and key uniqueness. The OOF `(group_id, segment_id)` set must **exactly equal** the canonical valid-label, active-78 candidate set. Compare the union and missing/extra keys, not just the intersection or row count.
3. Compare each OOF `fold` to the saved mapping. Require exactly one fold per point and trip, no training-trip overlap with that fold's validation trips, and metadata documenting how predictions were produced.
4. Verify every candidate of every held-out point is present. Incomplete key coverage is a rejection, not an invitation to inner-join the remaining rows and silently reduce the evaluation population.
5. Require finite scores in [0,1] and within-point sums of 1, allowing only a small numerical tolerance. Retain the original candidate count even for no-positive groups.
6. Require matching student-test candidate keys, with predictions for all candidate rows. Attach reference targets only afterwards for evaluation; do not expose them to inference or the UI.
7. Recompute metrics under the shared definitions. Reconcile counts, fold means/SD, pooled confusion matrices, point denominators and all saved timings before changing registry status to `trained`/ready.

Schema-valid scores alone do not prove OOF provenance: a final model's in-sample predictions cannot be relabeled as OOF by adding fold numbers.

## Why the original DT report cannot enter the common table unchanged

The [preserved component](../components/decision_tree/original/README.md) and [source manifest](../components/decision_tree/source_manifest.json) document five relevant differences:

| Original component | Shared current protocol |
|---|---|
| Nearest unique reference-route arc as a proxy label | Ordered route-position DP label |
| 20 midpoint-nearest candidates | Exact radius 210 m, nearest K=100 |
| Four-fold StratifiedGroupKFold | Existing saved five trip folds |
| Nine original predictor fields | Ordered 32 direction-enhanced fields |
| Original 80-trip training scope | User-requested 78-trip new-run scope |

Its covered-only CV and original scoring definitions also differ from the shared candidate-miss-inclusive point metrics. Keep the original source/report unchanged for provenance, but refit and evaluate the shared pipeline before claiming an apples-to-apples DT result. Renaming nine fields or filling 23 columns with zeros does not adapt an already fitted nine-input tree into a 32-input model.

## KNN pilot is not a final model

[The existing pilot summary](../results/knn/20261118_144236/summary.json) records 100 training groups and 50 validation groups per fold, 250 validation groups total. Its plots and pilot OOF file are useful implementation checks, not full-validation coverage. No final saved KNN pipeline has been verified. A final fitted bundle, complete shared OOF/test scores and compatible metrics are still required.

## Stacking handoff

Wait for LR, KNN, DT and XGBoost to pass the acceptance checks. Join base predictions by the exact candidate keys and identical fold IDs. Use OOF probabilities as combiner inputs, with any agreed geometric extras; never use each base model's own training predictions.

Keep the base-model column order fixed and saved. Train the combiner without student-test targets, refit final base pipelines on the permitted training set, then produce independent student-test base probabilities for combiner inference. Export all four final base models, every fitted preprocessing object, the combiner, feature order and normalization rules.

A second-level CV score on globally generated base OOF columns is not automatically a fully nested outer-CV estimate: some meta-training base predictions can have depended on the outer validation fold. Use nested cross-fitting for that stronger claim or describe the evaluation limitation explicitly. Preserve the same final-evaluator boundary.

When replacing any base model, candidate protocol or feature schema, regenerate that model's matching OOF/test scores and retrain the combiner. The old combiner must not silently consume new ranker scores.

## Shared scoring and API handoff

Use the definitions in the saved contract: Top-1 includes all valid points and candidate misses; log loss is conditional on the true candidate being present; precision/recall/Macro F1/confusion matrices use one chosen candidate per point. For HMM evaluation, score the Viterbi selected path and use actual posterior probabilities for decoder log loss. Clearly define directed continuity and what measured prediction time includes.

Report candidate-row accuracy separately if included in the requested accuracy metrics; do not present it as point-level route accuracy. Include the candidate miss rate, per-trip or sequence-level results, fit time and inference time, with N/A and a reason only when a metric genuinely does not apply.

Frontend work stays with Claude. The backend handoff is [API_CONTRACT.md](../API_CONTRACT.md); [13_backend_api.ipynb](../13_backend_api.ipynb) contains the canonical backend and executed in-process checks. It must specify trip listing, request/response schemas, model selection, errors, prediction geometry, metrics and CORS/LAN configuration. A registry file or precomputed demo data is not a completed live API integration.

Ground-truth targets stay in server-side evaluation metadata and must not enter model predictors or UI responses. The runtime route comes from actual model inference, not hardcoded expected trip answers. Final integration evidence includes all ten trip choices, a real request from the second laptop, receiving-server logs and successful API/UI screenshots.

See [PROJECT_CHECKLIST.md](../PROJECT_CHECKLIST.md) for submission/report evidence and unresolved requirements. No single-model performance gain or locally loadable bundle completes the five-model, report, public-repository and two-laptop demonstration requirements.


## Re-run evidence after a model becomes ready

[01_dataset_checks.ipynb](../01_dataset_checks.ipynb) already records actual raw-file inventory, dtypes, coordinate/time checks and weak network connectivity for all 80 training plus 10 student-test trips. Results are under [outputs/inventory](../outputs/inventory). The active-training exclusion flag is reported separately from raw availability.

Rerun [14_model_visualizations.ipynb](../14_model_visualizations.ipynb) after marking an integrated model `trained` and saving its required artifacts. It reads registry paths, rejects incomplete/schema-incompatible entries, excludes pilots, and produces actual CV/student-test/log-loss/confusion plots plus per-model predicted-route Folium maps. Eligibility and skipped reasons are saved in [model_visualization_status.json](../outputs/inventory/model_visualization_status.json). A PDF map screenshot remains an explicit submission task, separate from the generated HTML.
