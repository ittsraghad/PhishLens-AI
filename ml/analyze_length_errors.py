
import csv
import json
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

import joblib
import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "phishlens_tfidf_logreg.joblib"

REPORT_DIR = PROJECT_ROOT / "reports" / "model_evaluation"
REPORT_PATH = REPORT_DIR / "length_error_analysis.json"


# --------------------------------------------------
# 2. Load testing dataset
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
                raise ValueError("Empty email body found.")

            if label not in {"0", "1"}:
                raise ValueError(f"Invalid label: {label}")

            texts.append(body)
            labels.append(int(label))

    return texts, np.array(labels, dtype=np.int8)


# --------------------------------------------------
# 3. Analyze errors by email length
# --------------------------------------------------

def main():
    print("=" * 75)
    print("PhishLens AI - Advanced Length Error Analysis")
    print("=" * 75)

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

    # Build a table of predictions.
    results = pd.DataFrame({
        "email_length": [len(body) for body in X_test],
        "true_label": y_true,
        "predicted_label": y_pred,
    })

    # Determine whether each prediction is correct.
    results["is_error"] = (
        results["true_label"] != results["predicted_label"]
    )

    results["is_false_positive"] = (
        (results["true_label"] == 0)
        & (results["predicted_label"] == 1)
    )

    results["is_false_negative"] = (
        (results["true_label"] == 1)
        & (results["predicted_label"] == 0)
    )

    # Group emails by character length.
    bins = [0, 500, 1000, 2000, 5000, float("inf")]

    labels = [
        "1-500",
        "501-1000",
        "1001-2000",
        "2001-5000",
        "5001+",
    ]

    results["length_group"] = pd.cut(
        results["email_length"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    # --------------------------------------------------
    # 4. Calculate overall and class-specific error rates
    # --------------------------------------------------

    grouped = results.groupby("length_group", observed=False)

    rows = []

    for group_label, group in grouped:
        total_emails = len(group)

        # True label 0 = safe email.
        safe_emails = int((group["true_label"] == 0).sum())

        # True label 1 = phishing-labeled email category.
        malicious_emails = int((group["true_label"] == 1).sum())

        total_errors = int(group["is_error"].sum())

        false_positives = int(group["is_false_positive"].sum())
        false_negatives = int(group["is_false_negative"].sum())

        # Error rate among all emails in this length group.
        overall_error_rate = (
            total_errors / total_emails * 100
            if total_emails else None
        )

        # False positive rate among genuinely safe emails.
        false_positive_rate = (
            false_positives / safe_emails * 100
            if safe_emails else None
        )

        # False negative rate among genuinely malicious-labeled emails.
        false_negative_rate = (
            false_negatives / malicious_emails * 100
            if malicious_emails else None
        )

        rows.append({
            "length_group": str(group_label),
            "total_emails": total_emails,
            "safe_emails": safe_emails,
            "malicious_labeled_emails": malicious_emails,
            "total_errors": total_errors,
            "overall_error_rate_percent": overall_error_rate,
            "false_positives": false_positives,
            "false_positive_rate_percent": false_positive_rate,
            "false_negatives": false_negatives,
            "false_negative_rate_percent": false_negative_rate,
        })

    # --------------------------------------------------
    # 5. Save JSON report
    # --------------------------------------------------

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    total_errors = int(results["is_error"].sum())

    report = {
        "testing_rows": len(results),
        "total_errors": total_errors,
        "overall_error_rate_percent": float(
            results["is_error"].mean() * 100
        ),
        "length_groups": rows,
        "definitions": {
            "false_positive_rate": (
                "False positives divided by all genuinely safe emails "
                "in the same length group."
            ),
            "false_negative_rate": (
                "False negatives divided by all genuinely malicious-labeled "
                "emails in the same length group."
            ),
        },
        "note": (
            "Label 1 combines phishing, spam, and scam emails according "
            "to the source dataset. These results describe the test set "
            "and do not prove that email length causes classification errors."
        ),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    # --------------------------------------------------
    # 6. Print results
    # --------------------------------------------------

    print("\n[4/4] Analysis completed.\n")

    print("OVERALL ERROR RATE BY EMAIL LENGTH")
    print("-" * 55)
    print(f"{'Length':<12}{'Emails':>10}{'Errors':>10}{'Error %':>12}")

    for row in rows:
        rate = row["overall_error_rate_percent"]
        rate_text = f"{rate:.2f}%" if rate is not None else "N/A"

        print(
            f"{row['length_group']:<12}"
            f"{row['total_emails']:>10,}"
            f"{row['total_errors']:>10,}"
            f"{rate_text:>12}"
        )

    print("\nFALSE POSITIVE RATE AMONG SAFE EMAILS")
    print("-" * 55)
    print(f"{'Length':<12}{'Safe':>12}{'FP':>10}{'FP Rate':>12}")

    for row in rows:
        rate = row["false_positive_rate_percent"]
        rate_text = f"{rate:.2f}%" if rate is not None else "N/A"

        print(
            f"{row['length_group']:<12}"
            f"{row['safe_emails']:>12,}"
            f"{row['false_positives']:>10,}"
            f"{rate_text:>12}"
        )

    print("\nFALSE NEGATIVE RATE AMONG MALICIOUS-LABELED EMAILS")
    print("-" * 60)
    print(
        f"{'Length':<12}"
        f"{'Malicious':>14}"
        f"{'FN':>10}"
        f"{'FN Rate':>12}"
    )

    for row in rows:
        rate = row["false_negative_rate_percent"]
        rate_text = f"{rate:.2f}%" if rate is not None else "N/A"

        print(
            f"{row['length_group']:<12}"
            f"{row['malicious_labeled_emails']:>14,}"
            f"{row['false_negatives']:>10,}"
            f"{rate_text:>12}"
        )

    print(f"\nOverall error rate: {report['overall_error_rate_percent']:.2f}%")
    print(f"Report saved to: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
