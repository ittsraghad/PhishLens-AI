import csv
import sys
import json

from collections import Counter
from pathlib import Path


# Allow large email bodies inside CSV fields.
csv.field_size_limit(sys.maxsize)


# Locate project directories.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATASET_PATH = (
    PROJECT_ROOT / "data" / "raw" / "Merged_Dataset.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATASET_PATH = (
    PROCESSED_DIR / "prepared_emails.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"

REPORT_PATH = (
    REPORT_DIR / "preparation_report.json"
)


def prepare_dataset():
    """
    Validate and prepare email data without modifying the original file.
    """

    # Create output directories if they do not exist.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    empty_bodies = 0
    missing_labels = 0
    invalid_labels = 0

    removed_empty_bodies = 0
    removed_missing_labels = 0

    label_counts_before = Counter()
    label_counts_after = Counter()

    # Labels documented by the dataset authors.
    allowed_labels = {"0.0", "1.0"}

    print("Starting dataset preparation...")
    print(f"Source dataset: {RAW_DATASET_PATH}")

    if not RAW_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATASET_PATH}"
        )

    # Write to a temporary file first, so an interrupted run
    # does not leave a partially written final dataset.
    temporary_output_path = (
        PROCESSED_DIR / "prepared_emails.tmp.csv"
    )

    try:
        with RAW_DATASET_PATH.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as source_file:

            reader = csv.DictReader(source_file)

            expected_columns = {"body", "label"}
            actual_columns = set(reader.fieldnames or [])

            if not expected_columns.issubset(actual_columns):
                raise ValueError(
                    f"Unexpected dataset columns: {reader.fieldnames}"
                )

            with temporary_output_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as output_file:

                writer = csv.DictWriter(
                    output_file,
                    fieldnames=["body", "label"],
                )

                writer.writeheader()

                for row in reader:
                    total_rows += 1

                    body = row.get("body")
                    label = row.get("label")

                    # Check whether the email body is empty.
                    if body is None or not body.strip():
                        empty_bodies += 1
                        removed_empty_bodies += 1
                        continue

                    # Check whether the label is missing.
                    if label is None or not label.strip():
                        missing_labels += 1
                        removed_missing_labels += 1
                        continue

                    # Normalize whitespace around the label.
                    label = label.strip()

                    # Reject unexpected labels rather than guessing.
                    if label not in allowed_labels:
                        invalid_labels += 1
                        raise ValueError(
                            f"Unexpected label {label!r} "
                            f"at source data row {total_rows + 1}"
                        )

                    label_counts_before[label] += 1

                    # Preserve the email text, apart from whitespace
                    # around the beginning and end of the body.
                    cleaned_body = body.strip()

                    writer.writerow(
                        {
                            "body": cleaned_body,
                            "label": int(float(label)),
                        }
                    )

                    label_counts_after[label] += 1

        # Replace the final output only after the complete write succeeds.
        temporary_output_path.replace(PROCESSED_DATASET_PATH)

    finally:
        # Remove an incomplete temporary file if an error occurs.
        if temporary_output_path.exists():
            temporary_output_path.unlink()

    # Prepare a report describing what happened.
    report = {
        "source_dataset": str(RAW_DATASET_PATH.name),
        "processed_dataset": str(
            PROCESSED_DATASET_PATH.relative_to(PROJECT_ROOT)
        ),
        "total_source_rows": total_rows,
        "empty_bodies_removed": removed_empty_bodies,
        "missing_labels_removed": removed_missing_labels,
        "invalid_labels": invalid_labels,
        "rows_written": sum(label_counts_after.values()),
        "label_distribution_before": dict(
            sorted(label_counts_before.items())
        ),
        "label_distribution_after": dict(
            sorted(label_counts_after.items())
        ),
        "label_meanings": {
            "0": "Safe or legitimate email",
            "1": (
                "Dataset phishing label, including phishing, "
                "spam, and scam emails"
            ),
        },
        "duplicate_handling": (
            "Duplicates have not been removed. "
            "They require a separate documented decision."
        ),
        "original_dataset_modified": False,
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(
            report,
            report_file,
            indent=4,
            ensure_ascii=False,
        )

    # Display a summary.
    print("\n--- Dataset Preparation Results ---")
    print(f"Original rows: {total_rows:,}")
    print(f"Empty bodies removed: {removed_empty_bodies:,}")
    print(f"Missing labels removed: {removed_missing_labels:,}")
    print(f"Invalid labels: {invalid_labels:,}")
    print(
        f"Rows written: {sum(label_counts_after.values()):,}"
    )

    print("\nLabel distribution after preparation:")

    for label, count in sorted(label_counts_after.items()):
        print(f"  Label {label}: {count:,}")

    print(f"\nPrepared dataset: {PROCESSED_DATASET_PATH}")
    print(f"Preparation report: {REPORT_PATH}")
    print("\nOriginal dataset was not modified.")


if __name__ == "__main__":
    prepare_dataset()