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

The notebooks use NumPy, pandas, Matplotlib, Folium, Branca, pyproj,
Shapely 2, PyArrow, scikit-learn, Optuna, and XGBoost.

All models must reuse `outputs/folds/group_folds.csv` and
`outputs/folds/contract.json`. Split by trip, fit preprocessing within each
training split, retain candidate misses in the Top-1 denominator, and never
use student-test trips for tuning. See `AGENTS.md` for the planned models.

`Student folder/` contains the supplied training and student-test trips.
`PathSnap_student_materials/` contains the supplied PDF guidelines.
`outputs/` and `results/` contain the current generated artifacts and results.
