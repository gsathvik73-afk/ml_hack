# PathSnap ML project

Jupyter notebooks for GPS-to-road map matching, with the supplied datasets,
project guidelines, generated features, shared folds, and model results.

## Download the complete project

Install Git LFS before cloning so the large data files are downloaded:

```sh
git lfs install
git clone https://github.com/gsathvik73-afk/ml_hack.git
cd ml_hack
git lfs pull
```

## Notebook order

Run cells from top to bottom, starting from the project directory.

1. `01_eda.ipynb` — explore the training trips.
2. `02_geometry.ipynb` — inspect road geometry and nearby segments.
3. `03_label_points.ipynb` — align GPS points to reference routes.
4. `04_candidates.ipynb` — generate candidate road segments.
5. `05_feature_helpers.ipynb` — shared feature definitions loaded by notebooks.
6. `05_features.ipynb` — build candidate features.
7. `05_features_folds.ipynb` — create shared trip folds and scoring rules.
8. `06_model_helpers.ipynb` — shared definitions loaded by model notebooks.
9. `06a_xgb_ranker.ipynb` — tune and train the XGBoost ranker.
10. `07_student_test.ipynb` — preprocess student-test trips with frozen settings.
11. `07_bayesian_tuning.ipynb` — run 50 trials over eight parameters on the shared trip folds.
12. `07_ranker_comparison.ipynb` — compare untuned and tuned rankers on full-row CV and student-test trips.

13. `08_hmm_viterbi.ipynb` — tune decoder settings on training OOF predictions and decode student-test emissions.
14. `09_evaluate.ipynb` — compare available routes, continuity and timing; record the selected model.
15. `10_export.ipynb` — validate and export the selected model and frozen preprocessing/decoder settings.

The Step 7 search records whether it uses a fixed sample of complete GPS
groups or all rows. Final comparisons and refits use all eligible training
groups. Student-test scores are opened only after the chosen parameters
are frozen.

## Front-end demo

`12_demo_data.ipynb` precomputes the demo data for the 10 student-test trips
from the saved tuned XGBoost predictions and writes `frontend/data.js`. Open
`frontend/index.html` in a browser (or run `python3 -m http.server --directory frontend`).
It shows all trip metrics and replays the picked trip on a live map at a
chosen speed. Nothing is computed in the browser. The map uses Leaflet and
OpenStreetMap tiles from the internet.

Install the tested environment with `python3 -m pip install -r requirements.txt`.

The notebooks use NumPy, pandas, Matplotlib, Folium, Branca, pyproj,
Shapely 2, PyArrow, scikit-learn, Optuna, and XGBoost.

All models must reuse `outputs/folds/group_folds.csv` and
`outputs/folds/contract.json`. Split by trip, fit preprocessing within each
training split, retain candidate misses in the Top-1 denominator, and never
use student-test trips for tuning. See `AGENTS.md` for the planned models.

`Student folder/` contains the supplied training and student-test trips.
`PathSnap_student_materials/` contains the supplied PDF guidelines.
`outputs/` and `results/` contain the current generated artifacts and results.

## Current result

The latest direction-feature experiment is `11_direction_experiment.ipynb`.
It excludes training trips 33 and 54, adds eight deterministic features from
`05_direction_features.ipynb`, and reuses the saved folds and ranker settings.
With the same 78-trip population, pooled CV Top-1 improved from 89.66% to
90.67%; repeated student-test Top-1 improved from 89.43% to 89.89%.
The model and its required 32-column feature list are in
`results/direction_features/`. The earlier export described below is retained
as the original baseline; it does not contain this experimental feature set.

The 50-trial search used 10,000 complete GPS groups across all 80 training
trips. The final CV comparisons and model refits used all eligible training
groups. On the ten student-test trips, the tuned ranker scored 89.15% Top-1,
compared with 88.84% untuned and 88.60% with HMM/Viterbi. The fixed selection
rule chose the tuned ranker without HMM.

See `results/final_comparison.csv` and `results/final_selection.json` for
metrics and timing scope. `exports/final_model.zip` contains the validated
model, feature order and frozen preprocessing settings. This is an export
bundle; the API has not been deployed. The integrated backend now lives in `13_backend_api.ipynb`; see the current integration below for its status.


## Current shared-model integration

`models/registry.json` is the current model/artifact registry. XGBoost remains
the default at **90.67% pooled CV / 89.89% repeated student-test Top-1**.
Trips 33 and 54 remain excluded from training by user request, with raw data
and earlier 80-trip baselines retained. Student-test results have been viewed
repeatedly during development; an untouched evaluator score is not claimed.

- `01_dataset_checks.ipynb`: all 90 supplied trips, missingness and network checks.
- `06_shared_models.ipynb`: shared 32 features, trip folds, labels and scoring.
- `06_decision_tree.ipynb`: supplied tree settings adapted to the shared protocol;
  fast fitting uses up to 250 whole GPS groups per training trip, while all
  held-out points receive OOF predictions. This sample budget is recorded.
- `components/decision_tree/original/`: the five supplied files preserved unchanged.
- `06b_stacking.ipynb`: validated OOF alignment and ensemble workflow, waiting
  for the final logistic-regression and KNN models/complete shared-fold OOF.
  The existing KNN pilot is not a completed final model.
- `12_hmm_current.ipynb`: current ranker decoding; HMM reduces disconnected
  jumps but lowers student-test Top-1 to 89.18%, so it is optional.
- `13_backend_api.ipynb`: registry-based FastAPI with real inference checks.
- `14_model_visualizations.ipynb`: measured comparison plots and predicted-road map.
- `15_export_current.ipynb`: versioned current 32-feature ranker export.

Run `13_backend_api.ipynb` once to generate `.runtime/api.py`, then start:

```sh
python3 -m uvicorn api:app --app-dir .runtime --host 0.0.0.0 --port 8000
```

See `API_CONTRACT.md` for requests, environment variables and Claude's UI handoff.
The backend has passed in-process API tests; a two-laptop LAN demonstration
and its screenshots remain pending. Front-end work is owned by Claude.

`PROJECT_CHECKLIST.md` tracks the original requirements and actual evidence.
`docs/MODEL_INTEGRATION.md` explains teammate artifact compatibility. The final
five-model table, trained ensemble and consolidated final PDF require the
remaining model files and actual demonstration evidence.

The integrated decision tree completed at **88.22% pooled CV / 87.18% student-test Top-1**. Its full held-out OOF and final pipeline are saved under `results/decision_tree_shared/`; a real backend prediction also passed. XGBoost remains the active model.

Generated-data cleanup is recorded in `16_cleanup_generated.ipynb` and `outputs/inventory/redundant_cleanup.json`. Duplicate ranker tables, old feature caches and superseded large prediction files are removed; current32 artifacts and historical metric summaries remain. Historical experiment notebooks need their feature/prediction generation steps rerun to rebuild removed caches.
