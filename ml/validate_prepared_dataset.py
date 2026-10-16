import csv
import sys
from collections import Counter
from pathlib import Path


# Allow large email bodies inside CSV fields.
csv.field_size_limit(sys.maxsize)

# Locate the prepared dataset.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prepared_emails.csv"
)


def validate_prepared_dataset():
    print("Starting prepared dataset validation...")
    print(f"Dataset: {DATASET_PATH}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {DATASET_PATH}"
        )

    total_rows = 0
    empty_bodies = 0
    missing_labels = 0
    invalid_labels = 0
    label_counts = Counter()

    allowed_labels = {"0", "1"}

    with DATASET_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        expected_columns = {"body", "label"}
        actual_columns = set(reader.fieldnames or [])

        if not expected_columns.issubset(actual_columns):
            raise ValueError(
                f"Unexpected columns: {reader.fieldnames}"
            )

        for row in reader:
            total_rows += 1

            body = row.get("body")
            label = row.get("label")

            if body is None or not body.strip():
                empty_bodies += 1

            if label is None or not label.strip():
                missing_labels += 1
                continue

            label = label.strip()

            if label not in allowed_labels:
                invalid_labels += 1
                continue

            label_counts[label] += 1

    print("\n--- Validation Results ---")
    print(f"Total rows: {total_rows:,}")
    print(f"Empty email bodies: {empty_bodies:,}")
    print(f"Missing labels: {missing_labels:,}")
    print(f"Invalid labels: {invalid_labels:,}")

    print("\nLabel distribution:")

    for label in sorted(label_counts):
        count = label_counts[label]
        percentage = count / total_rows * 100 if total_rows else 0

        label_name = (
            "Safe"
            if label == "0"
            else "Phishing-labeled / malicious"
        )

        print(
            f"  {label_name} ({label}): "
            f"{count:,} ({percentage:.2f}%)"
        )

    print("\n--- Validation Summary ---")

    if (
        empty_bodies == 0
        and missing_labels == 0
        and invalid_labels == 0
        and total_rows == sum(label_counts.values())
        and total_rows > 0
    ):
        print("PASS: The prepared dataset passed these checks.")
    else:
        print("WARNING: Some validation checks need attention.")

    print("The dataset was read-only; no data was modified.")


if __name__ == "__main__":
    validate_prepared_dataset()