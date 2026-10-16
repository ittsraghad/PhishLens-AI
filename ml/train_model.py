
import csv
import json
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "phishlens_tfidf_logreg.joblib"

REPORT_DIR = PROJECT_ROOT / "reports" / "model_evaluation"
REPORT_PATH = REPORT_DIR / "baseline_metrics.json"


# --------------------------------------------------
# 2. Load and validate dataset
# --------------------------------------------------

def load_dataset(file_path):
    texts = []
    labels = []

    with open(file_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {"body", "label"}

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Dataset must contain 'body' and 'label' columns: {file_path}"
            )

        for row in reader:
            body = row["body"].strip()
            label = row["label"].strip()

            if not body:
                raise ValueError(f"Empty email body found in {file_path}")

            if label not in {"0", "1"}:
                raise ValueError(f"Invalid label '{label}' in {file_path}")

            texts.append(body)
            labels.append(int(label))

    return texts, np.array(labels, dtype=np.int8)


# --------------------------------------------------
# 3. Build the machine learning pipeline
# --------------------------------------------------

def build_model():
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=30000,
                ngram_range=(1, 2),
                min_df=2,
                stop_words="english",
                dtype=np.float32,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])

    return model


# --------------------------------------------------
# 4. Train and evaluate the model
# --------------------------------------------------

def main():
    print("=" * 60)
    print("PhishLens AI - Baseline Model Training")
    print("=" * 60)

    # Check that the required datasets exist.
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found: {TRAIN_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Testing dataset not found: {TEST_PATH}")

    # Load the training data.
    print("\n[1/5] Loading training dataset...")
    X_train, y_train = load_dataset(TRAIN_PATH)

    print(f"Training rows: {len(X_train):,}")
    print(f"Training safe emails (0): {int(np.sum(y_train == 0)):,}")
    print(f"Training phishing-labeled emails (1): {int(np.sum(y_train == 1)):,}")

    # Load the testing data.
    print("\n[2/5] Loading testing dataset...")
    X_test, y_test = load_dataset(TEST_PATH)

    print(f"Testing rows: {len(X_test):,}")

    # Build the model.
    print("\n[3/5] Building model...")
    model = build_model()

    # Fit TF-IDF and train the classifier using training data only.
    print("Training the model. This may take some time...")
    model.fit(X_train, y_train)

    print("Training completed.")

    # Evaluate on the separate testing dataset.
    print("\n[4/5] Evaluating model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Safe (0)", "Phishing-labeled (1)"],
        output_dict=True,
        zero_division=0,
    )

    print("\n--- Baseline Evaluation Results ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    # Prepare output folders.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the trained model.
    print("\n[5/5] Saving model and evaluation report...")
    joblib.dump(model, MODEL_PATH)

    # Save metrics in a JSON report.
    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "dataset": "Phishing-Email-Detection-Dataset",
        "positive_label": 1,
        "positive_label_meaning": (
            "Phishing-labeled emails, including spam and scam emails "
            "as grouped by the source dataset"
        ),
        "random_state": 42,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)

    print(f"\nSaved model to: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved evaluation report to: {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    print("\nBaseline training and evaluation completed.")


# --------------------------------------------------
# 5. Run the program
# --------------------------------------------------

if __name__ == "__main__":
    main()
