import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

csv.field_size_limit(sys.maxsize)


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = (
    PROJECT_ROOT / "data" / "processed" / "train_model.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT / "data" / "processed" / "validation.csv"
)

MODEL_PATH = (
    PROJECT_ROOT / "models" / "phishlens_threshold_experiment.joblib"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "model_evaluation"

REPORT_PATH = REPORT_DIR / "threshold_experiment.json"


# --------------------------------------------------
# 2. Experiment settings
# --------------------------------------------------

THRESHOLDS = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
]

RANDOM_STATE = 42


# --------------------------------------------------
# 3. Load and validate dataset
# --------------------------------------------------

def load_dataset(file_path):
    texts = []
    labels = []

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {"body", "label"}

        if not required_columns.issubset(
            reader.fieldnames or []
        ):
            raise ValueError(
                "Dataset must contain 'body' and 'label' "
                f"columns: {file_path}"
            )

        for row in reader:
            body = row["body"].strip()
            label = row["label"].strip()

            if not body:
                raise ValueError(
                    f"Empty email body found in {file_path}"
                )

            if label not in {"0", "1"}:
                raise ValueError(
                    f"Invalid label '{label}' in {file_path}"
                )

            texts.append(body)
            labels.append(int(label))

    if not texts:
        raise ValueError(
            f"Dataset is empty: {file_path}"
        )

    return texts, np.array(labels, dtype=np.int8)


# --------------------------------------------------
# 4. Build the machine learning model
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
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    return model


# --------------------------------------------------
# 5. Train the model and obtain validation probabilities
# --------------------------------------------------

def train_and_get_probabilities():

    print("\n[1/4] Loading training dataset...")

    X_train, y_train = load_dataset(TRAIN_PATH)

    print(f"Training rows: {len(X_train):,}")
    print(
        f"Safe emails (0): "
        f"{int(np.sum(y_train == 0)):,}"
    )
    print(
        f"Positive-label emails (1): "
        f"{int(np.sum(y_train == 1)):,}"
    )

    print("\n[2/4] Loading validation dataset...")

    X_validation, y_validation = load_dataset(
        VALIDATION_PATH
    )

    print(f"Validation rows: {len(X_validation):,}")

    if len(np.unique(y_train)) != 2:
        raise ValueError(
            "Training data must contain both labels: 0 and 1."
        )

    if len(np.unique(y_validation)) != 2:
        raise ValueError(
            "Validation data must contain both labels: 0 and 1."
        )

    print("\n[3/4] Building and training the model...")

    model = build_model()

    model.fit(X_train, y_train)

    print("Training completed.")

    print("\n[4/4] Calculating validation probabilities...")

    probabilities = model.predict_proba(X_validation)

    positive_class_index = int(
        np.where(model.classes_ == 1)[0][0]
    )

    positive_probabilities = probabilities[
        :, positive_class_index
    ]

    print("Validation probabilities calculated.")

    return (
        model,
        y_validation,
        positive_probabilities,
        len(X_train),
    )


# --------------------------------------------------
# 6. Evaluate different classification thresholds
# --------------------------------------------------

def evaluate_thresholds(y_true, probabilities):

    results = []

    for threshold in THRESHOLDS:

        y_pred = (
            probabilities >= threshold
        ).astype(np.int8)

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        precision = precision_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        )

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )

        (
            true_negatives,
            false_positives,
            false_negatives,
            true_positives,
        ) = matrix.ravel()

        results.append({
            "threshold": threshold,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "true_negatives": int(true_negatives),
            "false_positives": int(false_positives),
            "false_negatives": int(false_negatives),
            "true_positives": int(true_positives),
            "confusion_matrix": matrix.tolist(),
        })

    return results


# --------------------------------------------------
# 7. Save experiment results
# --------------------------------------------------

def main():

    print("=" * 60)
    print("PhishLens AI - Threshold Experiment")
    print("=" * 60)

    (
        model,
        y_validation,
        probabilities,
        training_rows,
    ) = train_and_get_probabilities()

    print("\nEvaluating classification thresholds...")

    results = evaluate_thresholds(
        y_validation,
        probabilities,
    )

    print("\n--- Threshold Comparison ---")

    print(
        f"{'Threshold':<12}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1-score':<12}"
        f"{'FP':<10}"
        f"{'FN':<10}"
    )

    for result in results:

        print(
            f"{result['threshold']:<12.2f}"
            f"{result['accuracy']:<12.4f}"
            f"{result['precision']:<12.4f}"
            f"{result['recall']:<12.4f}"
            f"{result['f1_score']:<12.4f}"
            f"{result['false_positives']:<10}"
            f"{result['false_negatives']:<10}"
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "experiment": "Classification threshold comparison",
        "model": "TF-IDF + Logistic Regression",
        "training_dataset": str(
            TRAIN_PATH.relative_to(PROJECT_ROOT)
        ),
        "validation_dataset": str(
            VALIDATION_PATH.relative_to(PROJECT_ROOT)
        ),
        "training_rows": int(training_rows),
        "validation_rows": int(len(y_validation)),
        "random_state": RANDOM_STATE,
        "thresholds": results,
        "selection_policy": (
            "No threshold is automatically selected. "
            "Compare validation metrics and error counts "
            "before choosing."
        ),
        "label_1_meaning": (
            "Phishing, spam, and scam emails grouped "
            "by the source dataset."
        ),
        "test_dataset_used_for_threshold_selection": False,
    }

    # Save the experimental model separately.
    joblib.dump(
        model,
        MODEL_PATH,
    )

    # Save threshold comparison results.
    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\nExperiment completed.")

    print(
        f"Model saved to: "
        f"{MODEL_PATH.relative_to(PROJECT_ROOT)}"
    )

    print(
        f"Report saved to: "
        f"{REPORT_PATH.relative_to(PROJECT_ROOT)}"
    )


# --------------------------------------------------
# 8. Run the experiment
# --------------------------------------------------

if __name__ == "__main__":
    main()
