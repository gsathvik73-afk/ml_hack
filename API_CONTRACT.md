# PathSnap backend API contract

This backend performs real candidate generation and model inference. It does not include a front end. Claude's UI can use the endpoints below.

The canonical code is `13_backend_api.ipynb`. Run it top to bottom to generate the ignored `.runtime/api.py` and execute in-process TestClient checks. Model paths and availability come from the server-controlled `models/registry.json`; no request accepts a filesystem path.

## Launch on laptop 1

The backend needs `fastapi`, `uvicorn`, `httpx` (for tests), and `joblib` alongside the existing ML dependencies. These are installed in the verified environment. From the project directory, after running the notebook:

```sh
PATHSNAP_PARTITION=student_test \
PATHSNAP_CORS_ORIGINS=http://localhost:5173,http://LAPTOP_2_IP:5173 \
python3 -m uvicorn --app-dir .runtime api:app --host 0.0.0.0 --port 8000
```

Replace `LAPTOP_2_IP` and the port with Claude's actual UI origin. Laptop 2 calls `http://LAPTOP_1_IP:8000`. CORS permits only the configured origins, GET/POST, and Content-Type; it does not enable browser credentials. Notebook verification uses ASGI TestClient, so it does not demonstrate reachability between the two physical laptops.

Environment variables are read at startup:

| Variable | Default | Meaning |
|---|---|---|
| `PATHSNAP_PROJECT_ROOT` | Parent of `.runtime` | Repository containing registry, models and helper notebooks |
| `PATHSNAP_DATA_ROOT` | `<project>/Student folder` | Raw data root controlled by the server operator |
| `PATHSNAP_PARTITION` | `student_test` | One of `train`, `student_test`, `evaluator_test`; cannot be overridden by a request |
| `PATHSNAP_CORS_ORIGINS` | localhost and 127.0.0.1 on ports 3000/5173 | Comma-separated exact browser origins |
| `PATHSNAP_ENABLE_METRICS` | `1` | Set `0` to disable all reference-metadata reads and aggregate scoring |

Regenerate `.runtime/api.py` after editing a tagged API cell. Restart Uvicorn after helper or environment changes. Registry updates and newly registered trained model files are read on subsequent requests without rebuilding the API. Model objects and four feature-prepared trips are cached per worker. Restart if raw trip files or frozen candidate/feature settings change.

## Endpoints

### GET `/health`

Returns `status` (`ok` or `degraded`), `active_model`, `ready_models`, `partition`, and `api_version`. Readiness checks registered model files; real inference is exercised separately by `/predict`.

### GET `/models`

Returns the active model ID and a `models` list. Each entry has `id`, `status`, `kind`, `ready`, and `decoder_available`. A teammate model still being integrated is listed but cannot be selected until its registry status is `trained` and its files exist.

Supported kinds:

- `xgboost_ranker`: native XGBoost JSON, per-point stable softmax.
- `sklearn_classifier`: estimator or joblib dictionary with `model` (a fitted Pipeline is preferred) and optional matching `feature_columns`. Normalize positive-class `predict_proba` within each point; zero sums fall back to uniform probabilities.
- `stacking`: joblib model/combiner, `base_model_order`, and a meta-feature list. Meta columns are the base model IDs plus `distance` and `candidate_count`/`n_candidates`. Base inference uses its own registered preprocessing. Logistic `decision_function` is normalized with per-point softmax. Missing base models produce a service-unavailable response rather than placeholder predictions.

### GET `/trips`

Returns `partition`, `count`, and `trips: [{"trip_id": "00000052"}, ...]`. Only eight-digit directories with `.nodes`, `.arcs`, and `.track` files are listed. Reference routes are not needed for listing or prediction.

### POST `/predict`

Request:

```json
{"trip_id": "00000052", "model": "xgboost", "decode": false}
```

`trip_id` is required and must be exactly eight ASCII digits. `model` defaults to the registry's active model. `decode` defaults to false. Other fields are rejected. With `decode=true`, the selected model must have a registered HMM configuration whose tuning source matches that model's registered OOF source. A different model's decoder is not silently reused.

Response fields:

| Field | Meaning |
|---|---|
| `trip_id`, `model`, `partition` | Requested trip and actual model/partition used |
| `decode_applied` | Whether Viterbi was applied |
| `probability_kind` | Per-point model emission, or HMM posterior when decoding |
| `point_count`, `candidate_rows` | Observations and actual candidate-pair rows |
| `gps` | One observed record per point: `point_id`, `lon`, `lat`, `time` |
| `predictions` | One record per point: `point_id`, predicted `segment_id`, its `probability`, and `coordinates` |
| `road_network` | GeoJSON FeatureCollection of unique candidate road segments, suitable for a road layer |
| `metrics` | Aggregate server-computed scores, or null if disabled/unavailable |
| `metrics_status` | `available` or `unavailable_or_disabled` |
| `timings` | Feature, model, decoder, evaluation, geometry and total server-processing seconds |

`coordinates` is `[[start_lon, start_lat], [end_lon, end_lat]]`, preserving arc direction. Road-network GeoJSON uses longitude/latitude as well. Segment IDs are zero-based row IDs in this trip's `.arcs`. Point IDs are zero-based GPS row IDs. Empty shortlists retain a prediction record with null segment/probability/coordinates. A singleton has probability 1. With Viterbi, the selected segment need not be the pointwise posterior argmax; its returned probability is that selected segment's marginal posterior.

The response contains no true arc IDs, reference routes, per-point labels, route positions, or reference-derived bad-GPS flags. The UI should draw the observed GPS and returned predicted road segments only. Aggregate confusion matrices and scores are permitted; they do not expose pointwise ground truth.

## Prediction and scoring rules

Candidates use frozen radius 210 m and K=100, deterministic distance/segment-ID ordering, WGS84 to per-trip UTM projection, the shared geometry/motion/context helpers, eight direction features and geometric distance ratio. Each model receives only its registry-ordered feature columns. The current XGBoost feature list has 32 columns. Normalized probabilities are not a claim of calibration. Whole trips are required because features use future GPS observations.

Inference reads only raw nodes/arcs/track. When metrics are enabled, scoring starts **after** prediction. For the original student-test partition, matching saved server-only point metadata is used. Otherwise the server may align a supplied `.route` with the frozen monotonic DP (150 m cost cap, training-derived 208.357058 m flag threshold). No thresholds are learned from a request. If reference data is unavailable, prediction still works and `metrics` is null.

Aggregate metrics include Top-1, candidate-row macro F1, class-1 precision/recall, a 2×2 confusion matrix, candidate coverage/misses, conditional log loss, and graph-disconnected/missing transitions. Top-1 includes valid points whose true segment is absent from the candidates. Log loss includes only valid points whose reference segment is present. Reference-flagged points are excluded only from scoring, never from inference. Directed reachability may include intermediate arcs; an adjacent-point transition is not called disconnected merely because it skips a road segment.

## Timing scope

`server_processing_s` includes candidate/features, model probability, optional decoder, optional aggregate evaluation and geometry assembly in the request. It excludes final JSON serialization, network transfer and browser rendering. Feature caching makes repeat requests faster; `features_s` reports the actual request's cache lookup or construction time. `model_s` includes per-point normalization; `decode_s` includes HMM preparation and Viterbi/posteriors. These are actual server measurements, not a claim about complete LAN/UI latency.

## Errors

| Status | Meaning |
|---|---|
| 404 | Unknown model ID or trip absent from the server's selected partition |
| 409 | Decoder absent, unavailable, or trained for a different registered model |
| 422 | Invalid trip/model format, extra request fields, or invalid GPS input |
| 503 | Registered model not trained/available, incompatible feature metadata, missing stack dependency, or registry temporarily unavailable |

Examples:

```sh
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/models
curl http://127.0.0.1:8000/trips
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"trip_id":"00000052","model":"xgboost","decode":false}'
```

The notebook tests the real active model, each additional ready registered model, optional current-model HMM, request validation, error paths, CORS, and recursively checks that truth fields are absent from prediction responses. It does not launch or test a physical two-laptop LAN deployment.
