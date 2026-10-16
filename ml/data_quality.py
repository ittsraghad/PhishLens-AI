
import csv
import sys
import hashlib
from collections import Counter
from pathlib import Path


# Allow CSV fields larger than Python's default limit.
csv.field_size_limit(sys.maxsize)

# Locate the project directory and raw dataset.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "Merged_Dataset.csv"


def inspect_data_quality():
    total_rows = 0

    empty_bodies = 0
    empty_labels = 0
    empty_body_and_label = 0

    label_counts = Counter()

    # Store hashes instead of email text to reduce memory usage.
    seen_body_hashes = set()
    duplicate_bodies = 0

    print("Starting data quality assessment...")
    print(f"Dataset: {DATASET_PATH}")

    with DATASET_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        # Verify that the required columns exist.
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

            # Identify missing or whitespace-only values.
            body_is_empty = body is None or not body.strip()
            label_is_empty = label is None or not label.strip()

            if body_is_empty:
                empty_bodies += 1

            if label_is_empty:
                empty_labels += 1

            if body_is_empty and label_is_empty:
                empty_body_and_label += 1

            # Count all available labels.
            if not label_is_empty:
                label_counts[label.strip()] += 1

            # Check exact duplicate email bodies.
            if not body_is_empty:
                normalized_body = body.strip()

                body_hash = hashlib.sha256(
                    normalized_body.encode("utf-8")
                ).hexdigest()

                if body_hash in seen_body_hashes:
                    duplicate_bodies += 1
                else:
                    seen_body_hashes.add(body_hash)

    # Display the results.
    print("\n--- Data Quality Results ---")
    print(f"Total rows: {total_rows:,}")

    print("\nMissing-value checks:")
    print(f"Empty email bodies: {empty_bodies:,}")
    print(f"Empty labels: {empty_labels:,}")
    print(
        "Rows with both empty body and label: "
        f"{empty_body_and_label:,}"
    )

    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        percentage = (
            count / total_rows * 100 if total_rows else 0
        )
        print(f"  Label {label!r}: {count:,} ({percentage:.2f}%)")

    print("\nDuplicate-text check:")
    print(
        "Repeated non-empty email bodies: "
        f"{duplicate_bodies:,}"
    )

    print("\nAssessment completed. No data was modified.")


if __name__ == "__main__":
    inspect_data_quality()
