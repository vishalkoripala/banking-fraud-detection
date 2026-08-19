"""
generate_explainability_report.py

Produces:
  outputs/global_feature_importance.png  -- what drives the model overall
  outputs/example_explanations.json       -- a worked local explanation for
                                              one true fraud + one true legit
                                              transaction from the test set

Run after train_final_model.py (needs model/fraud_model.joblib to exist).
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from explain import FraudExplainer

DATA_PATH = "data/creditcard.csv"
MODEL_PATH = "model/fraud_model.joblib"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    # Same split as train_final_model.py, so "test" here is the same untouched holdout
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = joblib.load(MODEL_PATH)
    explainer = FraudExplainer(pipeline, background_X=X_train.sample(2000, random_state=RANDOM_STATE))

    # --- Global importance ---
    print("Computing global feature importance on a 3,000-row test sample ...")
    sample = X_test.sample(min(3000, len(X_test)), random_state=RANDOM_STATE)
    importance = explainer.global_importance(sample)
    print(importance.head(10).to_string(index=False))

    top15 = importance.head(15).iloc[::-1]  # reverse so most important is on top of barh
    plt.figure(figsize=(7, 6))
    plt.barh(top15["feature"], top15["mean_abs_shap"], color="#2c5f8a")
    plt.xlabel("Mean |SHAP value| (impact on model output, log-odds)")
    plt.title("What drives fraud predictions — global feature importance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/global_feature_importance.png", dpi=150)
    plt.close()
    print(f"Saved {OUTPUT_DIR}/global_feature_importance.png")

    # --- Two worked local examples: one true fraud, one true legit ---
    fraud_idx = y_test[y_test == 1].index[0]
    legit_idx = y_test[y_test == 0].index[0]

    examples = {}
    for label, idx in [("true_fraud_example", fraud_idx), ("true_legit_example", legit_idx)]:
        row = X_test.loc[[idx]]
        proba = pipeline.predict_proba(row)[0, 1]
        top_factors = explainer.explain_instance(row, top_n=5)
        examples[label] = {
            "actual_class": "fraud" if y_test.loc[idx] == 1 else "legit",
            "predicted_fraud_probability": round(float(proba), 6),
            "top_factors": top_factors,
        }

    with open(f"{OUTPUT_DIR}/example_explanations.json", "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Saved {OUTPUT_DIR}/example_explanations.json")
    print(json.dumps(examples, indent=2))


if __name__ == "__main__":
    main()
