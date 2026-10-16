import csv
import sys
import hashlib
import json
import random

from collections import Counter, defaultdict
from pathlib import Path


csv.field_size_limit(sys.maxsize)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "prepared_emails.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = OUTPUT_DIR / "train.csv"
TEST_PATH = OUTPUT_DIR / "test.csv"

REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"

REPORT_PATH = REPORT_DIR / "split_report.json"

TEST_SIZE = 0.20
RANDOM_SEED = 42


def split_dataset():
    print("Starting group-aware dataset split...")
    print(f"Dataset: {DATASET_PATH}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {DATASET_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Store row counts by body hash and label, not email text.
    # This keeps memory use lower than loading every email.
    group_counts = defaultdict(Counter)

    total_rows = 0

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
                f"Unexpected dataset columns: {reader.fieldnames}"
            )

        for row in reader:
            total_rows += 1

            body = row.get("body")
            label = row.get("label")

            if body is None or not body.strip():
                raise ValueError(
                    f"Empty body found at data row {total_rows + 1}"
                )

            if label not in {"0", "1"}:
                raise ValueError(
                    f"Unexpected label {label!r} "
                    f"at data row {total_rows + 1}"
                )

            normalized_body = body.strip()

            body_hash = hashlib.sha256(
                normalized_body.encode("utf-8")
            ).hexdigest()

            group_counts[body_hash][label] += 1

    print(f"Rows read: {total_rows:,}")
    print(f"Unique email-text groups: {len(group_counts):,}")

    # A text group must have only one label.
    conflicting_groups = {
        body_hash: counts
        for body_hash, counts in group_counts.items()
        if len(counts) > 1
    }

    if conflicting_groups:
        raise ValueError(
            f"Found {len(conflicting_groups)} groups with "
            "conflicting labels. Resolve these before splitting."
        )

    # Group hashes by label for approximate stratification.
    groups_by_label = defaultdict(list)

    for body_hash, counts in group_counts.items():
        label = next(iter(counts))
        groups_by_label[label].append(body_hash)

    rng = random.Random(RANDOM_SEED)

    train_groups = set()
    test_groups = set()

    # Split unique text groups within each class.
    for label, hashes in groups_by_label.items():
        rng.shuffle(hashes)

        if len(hashes) < 2:
            raise ValueError(
                f"Not enough unique groups for label {label}"
            )

        test_group_count = round(len(hashes) * TEST_SIZE)

        # Keep at least one group in both train and test.
        test_group_count = max(
            1,
            min(test_group_count, len(hashes) - 1),
        )

        test_groups.update(hashes[:test_group_count])
        train_groups.update(hashes[test_group_count:])

    # Write each row to the split assigned to its text group.
    train_rows = 0
    test_rows = 0

    train_label_counts = Counter()
    test_label_counts = Counter()

    temporary_train = OUTPUT_DIR / "train.tmp.csv"
    temporary_test = OUTPUT_DIR / "test.tmp.csv"

    try:
        with DATASET_PATH.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as source_file:

            reader = csv.DictReader(source_file)

            with temporary_train.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as train_file, temporary_test.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as test_file:

                train_writer = csv.DictWriter(
                    train_file,
                    fieldnames=["body", "label"],
                )

                test_writer = csv.DictWriter(
                    test_file,
                    fieldnames=["body", "label"],
                )

                train_writer.writeheader()
                test_writer.writeheader()

                for row in reader:
                    body = row["body"].strip()
                    label = row["label"].strip()

                    body_hash = hashlib.sha256(
                        body.encode("utf-8")
                    ).hexdigest()

                    clean_row = {
                        "body": body,
                        "label": label,
                    }

                    if body_hash in test_groups:
                        test_writer.writerow(clean_row)
                        test_rows += 1
                        test_label_counts[label] += 1
                    else:
                        train_writer.writerow(clean_row)
                        train_rows += 1
                        train_label_counts[label] += 1

        # Replace final outputs only after successful writes.
        temporary_train.replace(TRAIN_PATH)
        temporary_test.replace(TEST_PATH)

    finally:
        if temporary_train.exists():
            temporary_train.unlink()

        if temporary_test.exists():
            temporary_test.unlink()

    # Verify that train and test contain disjoint body hashes.
    train_hashes = set()
    test_hashes = set()

    for path, destination in (
        (TRAIN_PATH, train_hashes),
        (TEST_PATH, test_hashes),
    ):
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:
                body_hash = hashlib.sha256(
                    row["body"].strip().encode("utf-8")
                ).hexdigest()

                destination.add(body_hash)

    overlap = train_hashes.intersection(test_hashes)

    if overlap:
        raise RuntimeError(
            f"Data leakage detected: {len(overlap)} "
            "email texts occur in both splits."
        )

    # Report both row-level and group-level split results.
    report = {
        "source_dataset": str(
            DATASET_PATH.relative_to(PROJECT_ROOT)
        ),
        "train_dataset": str(
            TRAIN_PATH.relative_to(PROJECT_ROOT)
        ),
        "test_dataset": str(
            TEST_PATH.relative_to(PROJECT_ROOT)
        ),
        "random_seed": RANDOM_SEED,
        "target_test_size": TEST_SIZE,
        "total_rows": total_rows,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_row_percentage": round(
            train_rows / total_rows * 100, 2
        ),
        "test_row_percentage": round(
            test_rows / total_rows * 100, 2
        ),
        "train_unique_text_groups": len(train_hashes),
        "test_unique_text_groups": len(test_hashes),
        "overlapping_text_groups": len(overlap),
        "train_label_counts": dict(sorted(train_label_counts.items())),
        "test_label_counts": dict(sorted(test_label_counts.items())),
        "conflicting_label_groups": len(conflicting_groups),
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

    print("\n--- Dataset Split Results ---")
    print(f"Training rows: {train_rows:,}")
    print(f"Testing rows: {test_rows:,}")
    print(f"Training unique texts: {len(train_hashes):,}")
    print(f"Testing unique texts: {len(test_hashes):,}")
    print(f"Overlapping text groups: {len(overlap)}")

    print("\nTraining label distribution:")
    for label, count in sorted(train_label_counts.items()):
        print(f"  Label {label}: {count:,}")

    print("\nTesting label distribution:")
    for label, count in sorted(test_label_counts.items()):
        print(f"  Label {label}: {count:,}")

    print(f"\nReport saved to: {REPORT_PATH}")
    print("Dataset splitting completed.")


if __name__ == "__main__":
    split_dataset()