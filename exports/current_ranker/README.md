# Current PathSnap ranker

This provisional bundle uses 32 ordered features and excludes training trips 00000033 and 00000054.
Load ranker.json with XGBRanker.load_model and select feature_columns.json in its exact order.
Generate candidates from raw nodes/arcs/track using candidate_settings.json, then use the included feature and direction notebooks plus preprocessing.json.
Apply stable per-point softmax. The optional current-model HMM is configured in hmm_config.json but enabled=false by default.
The helpers require only observed GPS and road information. Do not pass reference routes or point labels as predictor inputs.
Student-test Top-1 is recorded in manifest.json as development evidence, not an independent final evaluator score.
This is an artifact bundle; API integration is in repository notebook 13_backend_api.ipynb. No running server or front end is included.
