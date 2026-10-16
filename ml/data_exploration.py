import csv
import sys
from collections import Counter
from pathlib import Path


csv.field_size_limit(sys.maxsize)


# Locate the project directory and the raw dataset.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "Merged_Dataset.csv"


def inspect_dataset():
    label_counts = Counter()
    total_rows = 0
    empty_bodies = 0
    empty_labels = 0
    invalid_rows = 0

    print("Starting dataset inspection...")
    print(f"Dataset: {DATASET_PATH}")

    # Read the CSV incrementally instead of loading it all into memory.
    with DATASET_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        # Check that the dataset contains the expected columns.
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

            # Count rows with missing or whitespace-only email text.
            if body is None or not body.strip():
                empty_bodies += 1

            # Count rows with missing or whitespace-only labels.
            if label is None or not label.strip():
                empty_labels += 1
                continue

            # Track each label without assuming its meaning yet.
            label = label.strip()
            label_counts[label] += 1

            # Flag rows that do not contain the expected two fields.
            if body is None or label is None:
                invalid_rows += 1

    print("\n--- Dataset Inspection Results ---")
    print(f"Total data rows: {total_rows:,}")
    print(f"Number of labels: {len(label_counts)}")

    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        percentage = count / total_rows * 100 if total_rows else 0
        print(f"  Label {label!r}: {count:,} ({percentage:.2f}%)")

    print("\nBasic data quality checks:")
    print(f"Empty email bodies: {empty_bodies:,}")
    print(f"Empty labels: {empty_labels:,}")
    print(f"Structurally invalid rows: {invalid_rows:,}")

    print("\nInspection completed. The original dataset was not modified.")


if __name__ == "__main__":
    inspect_dataset()
