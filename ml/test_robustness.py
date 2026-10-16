import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# Allow reading emails with large text fields.
csv.field_size_limit(sys.maxsize)

# Project paths.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "phishlens_tfidf_logreg.joblib"
REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_evaluation"
    / "robustness_report.json"
)

RANDOM_STATE = 42
SAMPLE_SIZE = 5000


def load_test_data():
    """Load a reproducible sample of the existing test dataset."""
    data = pd.read_csv(TEST_PATH)

    required_columns = {"body", "label"}
    if not required_columns.issubset(data.columns):
        raise ValueError(
            f"Test dataset must contain these columns: {required_columns}"
        )

    data = data.dropna(subset=["body", "label"]).copy()
    data["body"] = data["body"].astype(str)

    # Use a fixed random seed so the sample can be reproduced.
    sample_size = min(SAMPLE_SIZE, len(data))
    return data.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)


def create_variants(text):
    """Create formatting variants without intentionally changing the words."""
    return {
        "original": text,
        "lowercase": text.lower(),
        "uppercase": text.upper(),
        "collapsed_whitespace": " ".join(text.split()),
    }


def main():
    print("Loading test data...")
    data = load_test_data()

    print(f"Sample size: {len(data):,}")
    print("Loading saved model...")
    model = joblib.load(MODEL_PATH)

    original_texts = data["body"].tolist()
    original_predictions = model.predict(original_texts)

    variants = {
        "lowercase": [],
        "uppercase": [],
        "collapsed_whitespace": [],
    }

    for text in original_texts:
        text_variants = create_variants(text)

        for variant_name in variants:
            variants[variant_name].append(
                text_variants[variant_name]
            )

    results = {}

    for variant_name, texts in variants.items():
        predictions = model.predict(texts)

        changed_predictions = int(
            np.sum(predictions != original_predictions)
        )

        results[variant_name] = {
            "sample_size": len(data),
            "changed_predictions": changed_predictions,
            "prediction_change_rate_percent": round(
                100 * changed_predictions / len(data), 4
            ),
        }

    # Also measure how often the original model was correct
    # compared with the dataset's existing labels.
    labels = data["label"].astype(int).to_numpy()
    original_accuracy = float(
        np.mean(original_predictions == labels)
    )

    report = {
        "model": "TF-IDF + Logistic Regression",
        "dataset": "Existing held-out test set",
        "sample_size": len(data),
        "random_state": RANDOM_STATE,
        "original_sample_accuracy_percent": round(
            100 * original_accuracy, 4
        ),
        "formatting_tests": results,
        "interpretation_note": (
            "Prediction change rates measure sensitivity to these specific "
            "text transformations. They do not prove real-world robustness "
            "or generalization to independent email sources. Lowercase and "
            "uppercase changes may be naturally handled by the vectorizer; "
            "whitespace changes may or may not alter tokenization."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nROBUSTNESS TEST RESULTS")
    print("-" * 55)
    print(f"Original sample accuracy: {report['original_sample_accuracy_percent']:.2f}%")

    for variant_name, metrics in results.items():
        print(
            f"{variant_name:25s} "
            f"Changed predictions: {metrics['changed_predictions']:5d} / "
            f"{metrics['sample_size']:5d} "
            f"({metrics['prediction_change_rate_percent']:.2f}%)"
        )

    print(f"\nReport saved to: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()