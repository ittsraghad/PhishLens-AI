
import csv
import json
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "phishlens_tfidf_logreg.joblib"

REPORT_DIR = PROJECT_ROOT / "reports" / "model_evaluation"
ERRORS_PATH = REPORT_DIR / "misclassified_emails.csv"
SUMMARY_PATH = REPORT_DIR / "error_analysis.json"


# --------------------------------------------------
# 2. Load testing data
# --------------------------------------------------

def load_test_dataset(file_path):
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
# 3. Analyze predictions and errors
# --------------------------------------------------

def main():
    print("=" * 60)
    print("PhishLens AI - Error Analysis")
    print("=" * 60)

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Testing dataset not found: {TEST_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    print("\n[1/4] Loading testing data...")
    X_test, y_true = load_test_dataset(TEST_PATH)
    print(f"Testing rows: {len(X_test):,}")

    print("\n[2/4] Loading trained model...")
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")

    print("\n[3/4] Generating predictions...")
    y_pred = model.predict(X_test)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    false_positives = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negatives = int(np.sum((y_true == 1) & (y_pred == 0)))
    total_errors = int(np.sum(y_true != y_pred))

    # --------------------------------------------------
    # 4. Analyze email lengths by prediction group
    # --------------------------------------------------

    true_negative_lengths = [
        len(body)
        for body, true_label, predicted_label in zip(X_test, y_true, y_pred)
        if true_label == 0 and predicted_label == 0
    ]

    true_positive_lengths = [
        len(body)
        for body, true_label, predicted_label in zip(X_test, y_true, y_pred)
        if true_label == 1 and predicted_label == 1
    ]

    false_positive_lengths = [
        len(body)
        for body, true_label, predicted_label in zip(X_test, y_true, y_pred)
        if true_label == 0 and predicted_label == 1
    ]

    false_negative_lengths = [
        len(body)
        for body, true_label, predicted_label in zip(X_test, y_true, y_pred)
        if true_label == 1 and predicted_label == 0
    ]

    correct_lengths = true_negative_lengths + true_positive_lengths

    length_analysis = {
        "correct_predictions_average_length": (
            float(np.mean(correct_lengths)) if correct_lengths else None
        ),
        "true_negative_average_length": (
            float(np.mean(true_negative_lengths))
            if true_negative_lengths else None
        ),
        "true_positive_average_length": (
            float(np.mean(true_positive_lengths))
            if true_positive_lengths else None
        ),
        "false_positive_average_length": (
            float(np.mean(false_positive_lengths))
            if false_positive_lengths else None
        ),
        "false_negative_average_length": (
            float(np.mean(false_negative_lengths))
            if false_negative_lengths else None
        ),
    }

    print("\nAverage email length by prediction group:")

    for group, average_length in length_analysis.items():
        if average_length is not None:
            print(f"{group}: {average_length:.1f} characters")
        else:
            print(f"{group}: No samples")

    # --------------------------------------------------
    # 5. Save misclassified emails
    # --------------------------------------------------

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # This file contains email text. Keep it private.
    with open(
        ERRORS_PATH,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "true_label",
                "predicted_label",
                "error_type",
                "email_length",
                "email_body",
            ],
        )

        writer.writeheader()

        for body, true_label, predicted_label in zip(
            X_test, y_true, y_pred
        ):
            if true_label == predicted_label:
                continue

            if true_label == 0 and predicted_label == 1:
                error_type = "False Positive"
            else:
                error_type = "False Negative"

            writer.writerow({
                "true_label": int(true_label),
                "predicted_label": int(predicted_label),
                "error_type": error_type,
                "email_length": len(body),
                "email_body": body,
            })

    # --------------------------------------------------
    # 6. Save summary report
    # --------------------------------------------------

    summary = {
        "testing_rows": len(X_test),
        "total_errors": total_errors,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "confusion_matrix": matrix.tolist(),
        "email_length_analysis": length_analysis,
        "note": (
            "Label 1 includes phishing, spam, and scam emails "
            "as grouped by the source dataset."
        ),
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    # --------------------------------------------------
    # 7. Print final results
    # --------------------------------------------------

    print("\n[4/4] Error analysis completed.")
    print(f"Total errors: {total_errors:,}")
    print(f"False positives: {false_positives:,}")
    print(f"False negatives: {false_negatives:,}")

    print(
        "\nMisclassified emails saved to: "
        f"{ERRORS_PATH.relative_to(PROJECT_ROOT)}"
    )

    print(
        "Error summary saved to: "
        f"{SUMMARY_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
