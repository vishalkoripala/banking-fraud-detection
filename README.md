# Banking Fraud Detection — Leakage-Safe Pipeline, Explainable, Deployable

A scikit-learn fraud detection pipeline for the ULB Credit Card Fraud
Detection dataset (284,807 transactions, 492 fraud, 0.173%), built around
three things that matter more than model choice on a problem like this:
**outlier handling that can't leak, evaluation that isn't fooled by
imbalance, and a served model that can explain every decision it makes.**

It runs as a real API: `POST /predict` returns a fraud probability *and*
the five features that drove it, in plain units.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Get the data (150MB, not bundled here — see Data below)
#    place it at data/creditcard.csv

# 2. Train + evaluate + persist the deployable model
python train_final_model.py

# 3. Generate explainability artifacts (global importance + worked examples)
python generate_explainability_report.py

# 4. Serve it
uvicorn api:app --host 0.0.0.0 --port 8000
# demo UI (click-through, no curl needed): http://localhost:8000/
# interactive API docs: http://localhost:8000/docs

# or, containerized:
docker compose up --build
```

## Live demo

`http://localhost:8000/` (the API's own root route, no separate frontend to
run) serves a self-contained demo page written for a genuinely general
audience — no ML background assumed. It frames the system as "AI Fraud
Guard checking a purchase," explains the class-imbalance problem with a
smoke-detector analogy and plain counting ("caught 79 of 98, wrongly
blamed only 16 of 56,864"), and lets you start from a real fraud or real
legitimate transaction from the test set.

You can also genuinely experiment: the dollar amount is editable, so you
can type in any amount and see whether it changes the guard's mind, with
the other 28 anonymized signals held fixed from that real example. This is
a deliberate scope limit, not a missing feature — V1–V28 are anonymized
PCA components with no interpretable real-world values, so there is no
honest UI for "type in your own V14"; Amount is the one field a person can
meaningfully set themselves. Try a $23 purchase pattern at $4,999 instead
of its original $23 and watch fraud probability shift from 0.6% to 2.7%,
with Amount itself appearing as a contributing factor — a genuine
demonstration that the model responds to amount, not just to canned
examples.

The verdict reads "🚨 This looks like a STOLEN card" / "✅ This looks like a
HONEST purchase," and the explanation of *why* uses plain phrases like
"this raised suspicion a lot" rather than feature codes or probability
jargon. Nothing technical is deleted, only resequenced: exact feature
names, raw contribution values, the calibration threshold, and model name
are all still real and still present, one click away behind a "for
technical reviewers" toggle. Nothing on the page is mocked; open browser
dev tools during a demo and the network tab shows the real call.

## Using the API

```bash
# sample_fraud_txn.json / sample_legit_txn.json are real, complete example
# payloads (all 30 fields) pulled from the held-out test set -- copy-pasteable.
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_fraud_txn.json
```

```json
{
  "fraud_probability": 1.0,
  "flagged": true,
  "threshold_used": 0.9999999157713859,
  "top_factors": [
    {"feature": "V14", "value": -6.1743, "contribution": 8.0815, "direction": "increases_risk"},
    {"feature": "V10", "value": -4.8811, "contribution": 6.4245, "direction": "increases_risk"},
    {"feature": "V17", "value": -6.5365, "contribution": 6.3217, "direction": "increases_risk"},
    {"feature": "V12", "value": -4.6864, "contribution": 4.6504, "direction": "increases_risk"},
    {"feature": "V16", "value": -4.8497, "contribution": 4.4742, "direction": "increases_risk"}
  ],
  "model_name": "logistic_regression",
  "scoring_time_ms": 15.02
}
```

That's a real response for a real fraud transaction from the held-out test
set. `top_factors` is a live SHAP explanation, not a canned message — see
**Explainability** below.

Other endpoints: `GET /health` (liveness), `GET /model/info` (deployed
model's identity + honest test-set metrics).

## Results

**Held-out test set (56,962 transactions, 98 fraud, never touched during
CV, model selection, or threshold tuning):**

| Metric | Value |
|---|---|
| PR-AUC (Average Precision) | 0.708 |
| ROC-AUC | 0.971 |
| Precision @ deployed threshold | 0.832 |
| Recall @ deployed threshold | 0.806 |

Full methodology and caveats in [`MODEL_CARD.md`](MODEL_CARD.md).

Separately, `run_pipeline.py` does a broader 5-fold CV comparison across
the full dataset (not the train/test-split deployment artifact) —
useful for exploring model choice, and where `outputs/pr_curve_comparison.png`
/ `outputs/roc_curve_comparison.png` come from. **The two scripts answer
different questions on purpose:** `run_pipeline.py` asks "which approach
looks best under cross-validation," `train_final_model.py` asks "what will
the thing I'm about to ship actually do on data it's never seen" — conflating
those two is a common and easy mistake, so this project keeps them
deliberately separate.

Interestingly, the two scripts don't always agree on a winner: on the full
dataset, gradient boosting edges out logistic regression; on the 80% train
split used for the actual deployment decision, logistic regression wins.
With only ~400–500 fraud examples total, that kind of split-to-split
variance is real, not a bug — which is itself a reason `train_final_model.py`
picks the winner programmatically from CV rather than a hardcoded choice.

## Explainability

Every `/predict` response includes the top 5 features behind that specific
score — exact Shapley values via `shap.LinearExplainer` (or
`shap.TreeExplainer` if a tree-based model is selected on a future retrain;
`explain.py` picks whichever matches the deployed model automatically).
Values are reported in original units (e.g. `Amount = $4,812.50`), not the
scaled/capped units the model actually consumes internally, paired with the
direction of the effect.

Global feature importance across many predictions —
`outputs/global_feature_importance.png` — shows `V4`, `V10`, `V14`, `V12`,
`V17` as the strongest overall drivers, consistent with patterns
independently reported elsewhere for this dataset. `MODEL_CARD.md` has the
important caveat: these are anonymized PCA components, so "V14 matters most"
doesn't map to a business reason the way it would on raw features.

## Two things this project is actually about

**1. ROC-AUC lies to you here; PR-AUC doesn't.**
Both model families post a ROC-AUC around 0.97 — that looks close to
solved. PR-AUC tells a different story (0.71–0.76 depending on split/model).
The gap is mechanical: ROC's false-positive-rate denominator is the entire
legitimate-transaction population, so a few thousand false alarms barely
move it. Precision divides by *flagged* transactions instead, where the
same false alarms are a much bigger share of a much smaller number.
Compare `outputs/roc_curve_comparison.png` against
`outputs/pr_curve_comparison.png` directly.

**2. `class_weight="balanced"` breaks the 0.5 threshold.**
The pipeline uses `class_weight="balanced"` to cope with the 0.17% fraud
rate rather than resampling (simpler, nothing to leak). Reweighting the
loss as if classes were ~50/50 also skews predicted probabilities away from
the true prior — the conventional 0.5 cutoff gives roughly **6% precision**
at 91% recall for logistic regression on this data, while the properly
tuned threshold gets ~83% precision at a similar recall. This is exactly
why `evaluate.best_f1_threshold` reads the threshold off the out-of-fold PR
curve instead of assuming 0.5, and why `api.py` always applies
`model_metadata.json`'s tuned threshold rather than a hardcoded one.

## Design decisions

- **Outlier capping targets `Amount` only, not V1–V28.** V1–V28 are already
  PCA components, and fraud is disproportionately found in their tails —
  capping them would clip the exact signal the model needs. `Amount` is raw
  and right-skewed (learned cap bounds land at roughly $[-209, 292]$ on the
  full dataset, IQR × 3), and capping it brings in 6.65% of transactions
  from a max of $25,691 down to $292, without touching the anonymized
  features.
- **Bounds are learned inside `.fit()`, never on the full dataset.** The
  `OutlierCapper` in `transformers.py` is a proper `BaseEstimator` /
  `TransformerMixin` wired into a `ColumnTransformer` inside a `Pipeline`.
  Refit per CV fold (or per train/test split), capping bounds, imputation
  medians, and scaler statistics are all recomputed on that fold/split's
  training rows only.
- **RobustScaler, not StandardScaler**, on every numeric group — median/IQR
  based, so it stays stable in the presence of exactly the outliers this
  project handles.
- **StratifiedKFold, not KFold**, and a **stratified train/test split** for
  the same reason: with only ~400–500 fraud rows total, unstratified
  splitting can easily starve a fold (or the test set) of positive examples.
- **Model choice is data-driven, not hardcoded.** `train_final_model.py`
  runs both candidate pipelines through CV on the training split and keeps
  whichever wins on PR-AUC — see Results above for why that matters.
- **Deployed artifact = exactly what was evaluated.** The model persisted
  to `model/fraud_model.joblib` is fit on the 80% train split and never
  retrained on more data afterward, so the test-set metrics in
  `model_metadata.json` are true of the literal artifact being served, not
  an estimate of a different, larger-trained model.
- **Explainability is model-family-aware, not hardcoded to one model type.**
  `FraudExplainer` picks `LinearExplainer` or `TreeExplainer` based on the
  actual fitted classifier, so a future retrain that picks the other model
  family doesn't silently break (or silently mis-explain) the API.
- **Median imputation is included but not load-bearing on this dataset** —
  the ULB CSV has zero missing values. It's kept as a defensive step since
  production fraud feeds routinely have gaps; pointed at messier real-world
  data, imputation is already wired in rather than bolted on later.

## Project structure

```
fraud_pipeline/
├── data/
│   └── creditcard.csv            # not shipped — see Data below
├── transformers.py                # OutlierCapper (custom sklearn transformer)
├── pipeline.py                     # ColumnTransformer + model pipelines
├── evaluate.py                       # CV scoring, PR/ROC curves, confusion matrices
├── explain.py                         # model-agnostic SHAP explainability
├── run_pipeline.py                  # exploratory full-dataset CV model comparison
├── train_final_model.py              # honest train/test split -> the deployed artifact
├── generate_explainability_report.py # global importance + worked local examples
├── api.py                             # FastAPI service
├── demo.py                            # self-contained demo UI, served at GET /
├── sample_fraud_txn.json              # real example payload (true fraud, test set)
├── sample_legit_txn.json              # real example payload (true legit, test set)
├── tests/
│   ├── test_pipeline.py              # unit tests (transformers, pipeline, evaluate)
│   └── test_api.py                     # API integration tests
├── model/                             # generated by train_final_model.py
│   ├── fraud_model.joblib
│   ├── model_metadata.json
│   └── background_sample.csv          # small sample for SHAP at serving time
├── outputs/                            # generated plots + reports
│   ├── pr_curve_comparison.png
│   ├── roc_curve_comparison.png
│   ├── confusion_matrix_*.png
│   ├── global_feature_importance.png
│   ├── example_explanations.json
│   └── metrics_report.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── MODEL_CARD.md                       # intended use, limitations, governance
└── README.md
```

## Data

Built and evaluated against the real dataset — `Time, V1..V28, Amount,
Class`, 284,807 rows, 492 fraud — not a synthetic stand-in. It's the
standard Kaggle "Credit Card Fraud Detection" dataset (ULB Machine Learning
Group), from Dal Pozzolo et al. Not bundled here since it's a ~150MB
third-party file; grab it from Kaggle (`mlg-ulb/creditcardfraud`) and drop
it at `data/creditcard.csv`. The code only cares about the schema, so any
CSV with those five column patterns drops in unmodified.

## Deployment

**Want a public URL, not just local Docker?** See
[`DEPLOYMENT.md`](DEPLOYMENT.md) for verified, step-by-step instructions
(Render is the recommended free option — no credit card, Docker-native,
zero code changes needed).

Locally:

```bash
docker compose up --build
```

or manually:

```bash
docker build -t fraud-api .
docker run -p 8000:8000 -v $(pwd)/logs:/app/logs fraud-api
```

The image copies in whatever's currently in `model/` at build time — retrain
(`python train_final_model.py`) and rebuild to deploy a new model; nothing
else needs to change. The container runs as a non-root user, exposes a
Docker `HEALTHCHECK` against `/health`, and every prediction is appended to
`logs/predictions.log` as one JSON line (mounted as a volume so it survives
container restarts) for drift monitoring and after-the-fact audits.

**Data note for anyone adapting this to a different schema:** this specific
dataset's `V1`-`V28` are already anonymized PCA components, so logging full
feature vectors carries no PII risk here. If you point this pipeline at
raw, non-anonymized transaction fields instead, redact or hash sensitive
fields before they reach `api.py`'s logger.

## Troubleshooting

**`'LogisticRegression' object has no attribute 'multi_class'` (or any similar
`AttributeError` from `/predict`)** — this is a scikit-learn cross-version
pickle problem, not a bug in your setup. `model/fraud_model.joblib` is a
fitted scikit-learn object, and scikit-learn does not guarantee pickle
compatibility across versions. `requirements.txt` now pins every package to
an exact version (previously some were loose `>=` constraints, which is
what caused this originally), so a clean `pip install -r requirements.txt`
should prevent it entirely. If you still hit it — e.g. you're reusing an
old virtual environment — the fix is:

```bash
pip install -r requirements.txt   # get the exact, tested versions
rm -rf model/                     # remove the possibly-stale model
python train_final_model.py       # rebuild it with what's actually installed
```

The server also checks this automatically now: if `model/model_metadata.json`'s
recorded `sklearn_version` doesn't match what's currently installed, you'll
see a clear warning in the startup logs telling you exactly this, rather
than a cryptic error on the first request.

## Testing

```bash
pytest tests/ -v
```

17 tests: `transformers.py`/`pipeline.py`/`evaluate.py` unit tests
(synthetic data, no artifacts required) plus `api.py` integration tests via
FastAPI's `TestClient` (needs `model/` populated — run `train_final_model.py`
first). Covers correct capping/scaling behavior, that bounds are frozen at
fit time, valid-probability outputs, and API input validation (missing
fields, wrong types, out-of-range values all return `422`, never a 500 or a
silent bad prediction).

## Model card

[`MODEL_CARD.md`](MODEL_CARD.md) — intended use and explicit out-of-scope
uses, evaluation methodology, limitations (temporal validity, what
"anonymized features" means for fairness auditing), and threshold
governance (the deployed threshold optimizes F1, a symmetric objective;
real fraud/friction cost trade-offs usually aren't symmetric, so that's
flagged as a business decision, not just an engineering default to leave
alone).

## Possible extensions

- **SMOTE/ADASYN** as a second imbalance strategy to compare against
  `class_weight="balanced"` (via `imblearn.pipeline.Pipeline`, so
  resampling only ever touches the training fold).
- **Business-cost-weighted thresholding** — replace F1-maximization with an
  actual cost matrix (a missed fraud and a false decline rarely cost the
  same) and pick the threshold that minimizes expected dollar loss.
- **Precision@K** (e.g. "if an analyst can review 500 flags a day, what
  recall do we get?") — often more operationally realistic than one
  F1-optimal global threshold.
- **Automated drift detection** on the live `fraud_probability` distribution
  and on individual feature distributions, rather than the manual review
  `MODEL_CARD.md` currently recommends.
- **A model registry** (e.g. MLflow) instead of a single `model/` directory,
  once there's more than one model version in play at a time.
