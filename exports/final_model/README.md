# PathSnap exported ranker

Load ranker.json with XGBRanker.load_model. Select predictor columns in the exact order in feature_columns.json.
Generate candidates with candidate_settings.json and features with 05_feature_helpers.ipynb; add distance_ratio from preprocessing.json.
Apply stable softmax separately per GPS group. Retain empty and singleton observations in the trip metadata.
If hmm_config.json has enabled=true, run the included 08_hmm_helpers.ipynb decoder with those frozen settings.
The feature context uses future points, so requests must provide a whole trip.

Only nodes, directed arcs and GPS observations are prediction inputs. No reference route or point labels are required.
This is an integration bundle, not a running API. Review manifest.json for validation, selection and timing scope.
