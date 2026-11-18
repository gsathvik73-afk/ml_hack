# PathSnap Decision Tree — Experiment and Evaluation Report

## Summary

The existing candidate-generation and geometric proxy-label methodology was
preserved. Hyperparameters were selected using only the 80 training trips with
four-fold `StratifiedGroupKFold`, grouped by trip. The ten `student_test` trips
were evaluated only after selection; no reserved evaluator-testing data was
read.

The tuned model won the training cross-validation objective by a small margin,
but did **not** improve the student development results over the original
settings refit on the same full training data. The saved model is the model
selected by training-only CV; the development results are reported as-is and
were not used to select a replacement.

## 1. Model architecture

For each GPS observation, a midpoint KD-tree retrieves up to `k=20` road arcs.
Exact point-to-segment distances and motion features are computed for those
arcs. A binary `DecisionTreeClassifier` assigns each candidate a positive-class
probability. The classifier scores candidates; it does not decode a complete
route. The team HMM/Viterbi stage remains responsible for sequence decoding.

The final tree uses entropy, depth 8, minimum split size 100, minimum leaf size
10, `max_features=0.8`, balanced class weights, and `ccp_alpha=0.00001`.
Probabilities remain available through `predict_proba`.

## 2. Dataset description and handling

| Partition | Trips | Candidate rows | GPS observations |
|---|---:|---:|---:|
| Training | 80 | 4,168,340 | 208,417 |
| Student development | 10 | 383,200 | 19,160 |

Of the training observations, 192,597 had a positive candidate in the `k=20`
shortlist; 15,820 were candidate misses and were excluded from fitting and
covered-point CV ranking. All candidate rows, including misses, remain in the
development candidate-row evaluation. The development shortlist covered
18,489 of 19,160 observations (96.50%); 671 observations had no positive
candidate.

The project reads only `train/` and `student_test/`. Hyperparameter search and
the original-settings CV baseline use training trips only. The development
partition is evaluated after model selection.

## 3. Candidate features and proxy labels

The model's feature order is saved in `decision_tree_model.joblib` and is:

1. `distance_m`: exact point-to-segment distance in projected metres.
2. `distance_start_m`, `distance_end_m`: distances to the segment endpoints.
3. `segment_length_m`: candidate road-segment length.
4. `heading_difference_deg`: GPS movement bearing vs. candidate arc bearing.
5. `gps_speed_mps`: GPS displacement divided by the elapsed time.
6. `time_gap_s`: elapsed time used for the movement estimate.
7. `projection_fraction`: clamped projection position along the candidate.
8. `candidate_rank`: rank by exact point-to-segment distance.

GPS and road coordinates are projected into a local UTM CRS per trip. The
candidate shortlist uses nearest arc midpoints, followed by exact
point-to-segment geometry for those candidates.

**The label is not ground truth.** For each GPS point, the code finds the
geometrically nearest arc among the unique arcs listed in that trip's `.route`
file. Candidate rows matching that arc are labeled positive. The supplied
`.route` files do not provide a GPS-row-to-route-row alignment, so the scores
measure association with this geometric proxy, not definitive route-matching
accuracy. Candidate misses also show that the midpoint shortlist can omit the
proxy-positive arc.

## 4. Hyperparameter experiments

`RandomizedSearchCV` evaluated 24 randomly selected settings from these
search-space values:

| Parameter | Values tested |
|---|---|
| `criterion` | `gini`, `entropy` |
| `max_depth` | 8, 12, 16, 24, unlimited |
| `min_samples_split` | 20, 50, 100 |
| `min_samples_leaf` | 5, 10, 20 |
| `max_features` | all, `sqrt`, 0.8 |
| `class_weight` | none, `balanced`, `{0: 1, 1: 2}` |
| `ccp_alpha` | 0, 0.00001, 0.0001, 0.001 |

The search used a reproducible sample of up to 250 covered GPS observations per
training trip (400,000 candidate rows total), retaining every candidate for
each sampled observation. Four-fold `StratifiedGroupKFold` kept all rows from
each trip in one fold. The refit/selection metric was top-1 proxy accuracy on
covered points; candidate-row accuracy, precision, recall, and F1 were
recorded as secondary metrics. Ties in candidate probability were resolved by
the smallest `candidate_rank`.

| CV metric (mean ± std across folds) | Original settings | CV-selected settings |
|---|---:|---:|
| Top-1 proxy accuracy, covered points | 81.89% ± 1.45 pp | 82.68% ± 1.55 pp |
| Candidate-row accuracy | 95.84% ± 0.18 pp | 95.27% ± 0.54 pp |
| Candidate-row precision | 54.96% ± 1.14 pp | 51.66% ± 2.95 pp |
| Candidate-row recall | 93.16% ± 1.42 pp | 94.73% ± 1.99 pp |
| Candidate-row F1 | 69.12% ± 0.92 pp | 66.78% ± 2.15 pp |

The original-settings CV reference used the same folds and sampled training
rows. The CV-selected settings improved the selection metric by about 0.80
percentage points, but the CV secondary candidate-row metrics were lower.
`artifacts/decision_tree_cv_results.csv` contains the per-trial search results;
`artifacts/decision_tree_metrics.json` contains fold-level metrics and standard
deviations.

## 5. Original vs. tuned model on student development data

Both models below were refit on all covered training rows using their respective
settings. The original hyperparameters are therefore compared with the
CV-selected hyperparameters under the same training-data scope.

| Model | Candidate-row accuracy | Precision | Recall | F1 | Top-1 proxy accuracy, covered points | Top-1 proxy accuracy, all points |
|---|---:|---:|---:|---:|---:|---:|
| Original settings | 96.08% | 55.34% | 96.63% | 70.38% | 83.55% | 80.63% |
| CV-selected settings | 95.87% | 54.03% | 96.54% | 69.29% | 83.25% | 80.34% |

The tuned model's covered-point top-1 proxy score was 0.30 percentage points
lower on development data, and its candidate-row F1 was 1.09 percentage points
lower. This is not a development-set improvement. The model remains CV-selected
to avoid choosing hyperparameters against the development set.

Candidate-row confusion matrices use label order `[0, 1]` and include
observations whose proxy-positive arc was missed by the shortlist:

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Original settings | 350,295 | 14,416 | 623 | 17,866 |
| CV-selected settings | 349,527 | 15,184 | 640 | 17,849 |

“Top-1, covered points” excludes the 671 development candidate misses;
“top-1, all points” counts them as unsuccessful. Candidate-row accuracy is
affected by the large negative class and should not be read alone.

The earlier run's reported 80.56% top-1 value was labeled as a covered-point
score but included all 19,160 observations. The corrected evaluation now
reports covered-only and all-observation scores separately. This report's
original-vs-tuned table uses the newly refit original settings, not that older
partial-training model.

## 6. Limitations and interpretation

- The nearest-reference-route-arc label is a geometric proxy. There is no
  observation-to-route alignment in the supplied files.
- The model can only rank retrieved candidates. The development candidate
  coverage rate was 96.50%; a missed proxy-positive arc cannot be recovered by
  the classifier or sequence decoder.
- Per-candidate binary probabilities are not automatically a normalized
  distribution over a GPS observation's candidates.
- A CV winner can perform worse on the small development partition, as occurred
  here. The selected model should not be described as having improved
  development accuracy.
- Validate the label-association assumption with the PathSnap team before
  treating any reported score as real-world route-matching accuracy.

## 7. Integration with the main PathSnap project

The model bundle stores the fitted estimator, candidate `k`, feature names and
order, selected hyperparameters, and label metadata. The integration helper
returns one positive-class probability per input row and preserves
`trip_id`, `gps_index`, and `arc_id`:

```python
from pathlib import Path
import joblib
from decision_tree import build_trip_examples, score_candidates

bundle = joblib.load("artifacts/decision_tree_model.joblib")
candidates, _ = build_trip_examples(Path("path/to/trip"), bundle["candidate_k"])
candidate_scores = score_candidates(bundle, candidates)
```

Pass `candidate_scores` to the team's decoder without losing row-to-arc
identity. The output field is `candidate_probability`. If the HMM interface
requires normalized categorical emissions or log emissions, perform that
conversion according to the team's contract. This starter contains no HMM or
Viterbi implementation, so the final decoder contract still needs team
validation.

## Generated artifacts

- `artifacts/decision_tree_model.joblib` — CV-selected model and feature metadata.
- `artifacts/decision_tree_model_original.joblib` — original settings refit on all
  covered training rows.
- `artifacts/decision_tree_metrics.json` — metrics, CV results, and comparison.
- `artifacts/decision_tree_cv_results.csv` — randomized-search trials.
- `artifacts/decision_tree_visualization.png` and
  `artifacts/decision_tree_visualization.svg` — visualization of the top tree
  levels.
- `artifacts/decision_tree_rules.txt` — readable rules for the displayed levels.
- `artifacts/student_candidate_scores.csv` — probabilities with candidate IDs.
- `artifacts/train_candidate_features.csv`,
  `artifacts/student_candidate_features.csv`, and both trip diagnostics CSVs.
