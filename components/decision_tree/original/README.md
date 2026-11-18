# PathSnap — Decision Tree component

## What this component does

For each GPS observation, the pipeline retrieves the `k` nearest road-arc
midpoints, computes exact point-to-segment and motion features, and trains a
binary Decision Tree that scores each candidate arc. The output is a per-candidate
positive-class probability for the team's HMM/Viterbi sequence decoder.

## Dataset layout

```text
data/Student folder/
  train/<trip_id>/<trip_id>.track
  train/<trip_id>/<trip_id>.nodes
  train/<trip_id>/<trip_id>.arcs
  train/<trip_id>/<trip_id>.route
  student_test/<trip_id>/...
```

Only `train/` is used for model fitting and hyperparameter selection.
`student_test/` is held back until after model selection for development
evaluation. Reserved evaluator-testing data is not read by this component.

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

From the project root:

```powershell
python decision_tree.py --data "data/Student folder" --k 20
```

By default, the run performs 24 RandomizedSearchCV trials using 4-fold
StratifiedGroupKFold. Folds are grouped by trip ID, and the search sample keeps
all candidate rows for each sampled GPS observation (up to 250 observations
per trip). The selected model is then refit on all covered training candidate
rows. To control run time, use `--search-iterations`, `--cv-folds`,
`--cv-points-per-trip`, and `--n-jobs`. To rerun tuning against unchanged
candidate CSVs, use `--reuse-features`; the script checks the candidate `k`,
source trip IDs, and input-file timestamps before reusing them.

The search compares Gini and entropy, tree depth, split/leaf sizes, max features,
class weights, and cost-complexity pruning (`ccp_alpha`). Selection optimizes
top-1 proxy-label accuracy on covered CV points; candidate-row accuracy,
precision, recall, and F1 are also reported.

## First-evaluation metrics

After candidate features and the selected model exist, run the observation-level
evaluation:

```powershell
python evaluate_decision_tree.py --data "data/Student folder" --output artifacts
```

It reports per-GPS top-1 results, trip-qualified arc macro F1, normalized
candidate log loss, and training-versus-grouped-validation results. It updates
`artifacts/decision_tree_metrics.json` and writes
`artifacts/decision_tree_results.csv` and
`artifacts/decision_tree_evaluation_report.txt`. Validation uses the training
trips only; `student_test/` is evaluated after model selection.

## Feature engineering and labels

Features, in model input order:

1. `distance_m` — exact point-to-segment distance in metres.
2. `distance_start_m` and `distance_end_m` — point-to-endpoint distances.
3. `segment_length_m` — candidate arc length.
4. `heading_difference_deg` — difference between GPS movement bearing and arc bearing.
5. `gps_speed_mps` — GPS displacement divided by elapsed time.
6. `time_gap_s` — elapsed time used for the GPS movement estimate.
7. `projection_fraction` — clamped projection location along the candidate arc.
8. `candidate_rank` — exact point-to-segment distance rank among shortlisted arcs.

The `.route` file is a chronological list of arc IDs, not an observation-to-route
alignment. To preserve the starter's methodology, each GPS observation is labeled
with the geometrically nearest arc among that trip's unique reference-route arcs;
a candidate is positive if its arc ID matches. Covered observations are used for
training and CV. Candidate misses are retained in diagnostics and development
coverage metrics. These labels and all scores remain proxies, not definitive
GPS-row ground truth.

## Outputs

The `artifacts/` directory contains:

- `decision_tree_model.joblib` — fitted model, feature order, `k`, selected
  hyperparameters, and label metadata.
- `decision_tree_model_original.joblib` — original settings refit on the same
  covered training rows, retained for the comparison.
- `decision_tree_metrics.json` — CV and development metrics, confusion matrices,
  candidate coverage, and original-vs-tuned comparison.
- `decision_tree_cv_results.csv` — metrics and parameters for each search trial.
- `decision_tree_visualization.png` / `.svg` — readable visualization of the
  top tree levels; `decision_tree_rules.txt` provides the corresponding rules.
- `train_candidate_features.csv` / `student_candidate_features.csv` — generated
  feature rows and proxy labels.
- `student_candidate_scores.csv` — per-candidate positive probabilities with
  `trip_id`, `gps_index`, and `arc_id` preserved for integration.
- `train_trip_diagnostics.csv` / `student_trip_diagnostics.csv` — trip and
  candidate-miss diagnostics.

## HMM/Viterbi integration

Use the same candidate-generation function and pass the produced rows to
`score_candidates`. It returns one probability per candidate in input order,
alongside the trip, GPS-observation, and arc IDs:

```python
from pathlib import Path
import joblib
from decision_tree import build_trip_examples, score_candidates

bundle = joblib.load("artifacts/decision_tree_model.joblib")
candidates, _ = build_trip_examples(Path("path/to/trip"), bundle["candidate_k"])
scored = score_candidates(bundle, candidates)
```

`candidate_probability` is the binary model's positive-class probability; it is
not normalized across a GPS observation's candidate set. Preserve the IDs when
passing scores into the sequence decoder. If the HMM API specifically requires a
normalized categorical emission distribution, normalize within each
`(trip_id, gps_index)` group at that integration boundary.

See [DECISION_TREE_REPORT.md](./DECISION_TREE_REPORT.md) for the recorded
experiments, comparison, and evaluation results.
