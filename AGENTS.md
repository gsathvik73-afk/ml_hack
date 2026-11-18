# Coding preference

Keep all code as simple as possible. Avoid overcomplicating things or adding unnecessary abstractions, dependencies, or features. Prefer straightforward, readable solutions.

Write all project code in Jupyter notebooks (`.ipynb`). Use simple, readable cells in a clear execution order, with brief Markdown explanations where helpful.

# Planned models

Use logistic regression, KNN, a decision tree, XGBoost ranker, and a stacking ensemble in later modelling notebooks.

For the XGBoost ranker, include training, Optuna tuning, and a softmax over candidate scores for each GPS point to obtain HMM probabilities.

Build the stacking ensemble after logistic regression, KNN, the decision tree, and XGBoost are trained. Combine their predictions, using out-of-fold predictions grouped by trip to train the stacker.

All models must reuse `outputs/folds/group_folds.csv` and the OOF and metric rules in `outputs/folds/contract.json`. Split by trip, fit preprocessing inside each training split, and retain candidate misses in the Top-1 denominator. Never use student-test trips for tuning.
