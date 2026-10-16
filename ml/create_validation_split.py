
import csv
import sys
import hashlib
import json
import random

from collections import Counter, defaultdict
from pathlib import Path


# Allow large email bodies inside CSV fields.
csv.field_size_limit(sys.maxsize)


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "train.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

NEW_TRAIN_PATH = OUTPUT_DIR / "train_model.csv"
VALIDATION_PATH = OUTPUT_DIR / "validation.csv"

REPORT_DIR = (
    PROJECT_ROOT / "reports" / "data_quality"
)

REPORT_PATH = (
    REPORT_DIR / "validation_split_report.json"
)

VALIDATION_SIZE = 0.20
RANDOM_SEED = 42


# --------------------------------------------------
# 2. Helper functions
# --------------------------------------------------

def normalize_body(body):
    """
    Normalize a message for exact-duplicate grouping.

    This matches the normalization used by the existing
    split script: strip leading and trailing whitespace.
    """
    return body.strip()


def hash_body(body):
    """
    Return a SHA-256 fingerprint of the normalized email.
    """
    normalized_body = normalize_body(body)

    return hashlib.sha256(
        normalized_body.encode("utf-8")
    ).hexdigest()


def load_group_counts(file_path):
    """
    Count rows by unique email-text group and label.

    Email bodies are not retained in memory during this pass.
    """
    group_counts = defaultdict(Counter)
    total_rows = 0

    with file_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {"body", "label"}

        if not required_columns.issubset(
            set(reader.fieldnames or [])
        ):
            raise ValueError(
                "The source CSV must contain body and label columns."
            )

        for row in reader:
            total_rows += 1

            body = row.get("body")
            label = row.get("label")

            if body is None or not body.strip():
                raise ValueError(
                    f"Empty email body at data row {total_rows + 1}"
                )

            if label not in {"0", "1"}:
                raise ValueError(
                    f"Invalid label {label!r} "
                    f"at data row {total_rows + 1}"
                )

            body_hash = hash_body(body)

            group_counts[body_hash][label] += 1

    return group_counts, total_rows


# --------------------------------------------------
# 3. Create a group-aware validation split
# --------------------------------------------------

def create_validation_split():

    print("=" * 65)
    print("PhishLens AI - Validation Dataset Split")
    print("=" * 65)

    # Check that the source exists.
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {SOURCE_PATH}"
        )

    # Avoid accidentally overwriting existing outputs.
    if NEW_TRAIN_PATH.exists():
        raise FileExistsError(
            f"Output already exists: {NEW_TRAIN_PATH}\n"
            "Review or rename it before running this script."
        )

    if VALIDATION_PATH.exists():
        raise FileExistsError(
            f"Output already exists: {VALIDATION_PATH}\n"
            "Review or rename it before running this script."
        )

    if REPORT_PATH.exists():
        raise FileExistsError(
            f"Report already exists: {REPORT_PATH}\n"
            "Review or rename it before running this script."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Pass 1: Count duplicate groups without retaining
    # all email bodies in memory.
    # --------------------------------------------------

    print("\n[1/5] Inspecting source dataset...")

    group_counts, total_rows = load_group_counts(
        SOURCE_PATH
    )

    print(f"Source rows: {total_rows:,}")
    print(f"Unique text groups: {len(group_counts):,}")

    # Each exact text group must have one consistent label.
    conflicting_groups = sum(
        1
        for counts in group_counts.values()
        if len(counts) != 1
    )

    if conflicting_groups:
        raise ValueError(
            f"Found {conflicting_groups:,} text groups "
            "with conflicting labels. No split was created."
        )

    # --------------------------------------------------
    # Pass 2: Group unique texts by their label.
    # --------------------------------------------------

    groups_by_label = defaultdict(list)

    for body_hash, counts in group_counts.items():
        label = next(iter(counts))
        groups_by_label[label].append(body_hash)

    rng = random.Random(RANDOM_SEED)

    model_train_groups = set()
    validation_groups = set()

    print("\n[2/5] Assigning unique text groups...")

    for label in sorted(groups_by_label):

        hashes = groups_by_label[label].copy()
        rng.shuffle(hashes)

        if len(hashes) < 2:
            raise ValueError(
                f"Not enough unique groups for label {label}."
            )

        validation_group_count = round(
            len(hashes) * VALIDATION_SIZE
        )

        # Keep at least one group in each partition.
        validation_group_count = max(
            1,
            min(
                validation_group_count,
                len(hashes) - 1,
            ),
        )

        validation_groups.update(
            hashes[:validation_group_count]
        )

        model_train_groups.update(
            hashes[validation_group_count:]
        )

    if model_train_groups.intersection(validation_groups):
        raise RuntimeError(
            "Text-group overlap detected before writing files."
        )

    print(
        f"Training groups: {len(model_train_groups):,}"
    )
    print(
        f"Validation groups: {len(validation_groups):,}"
    )

    # --------------------------------------------------
    # Pass 3: Write new CSV files using temporary files.
    # The original train.csv is never modified.
    # --------------------------------------------------

    print("\n[3/5] Writing new datasets...")

    temporary_train_path = (
        OUTPUT_DIR / "train_model.tmp.csv"
    )

    temporary_validation_path = (
        OUTPUT_DIR / "validation.tmp.csv"
    )

    train_rows = 0
    validation_rows = 0

    train_label_counts = Counter()
    validation_label_counts = Counter()

    try:
        with SOURCE_PATH.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as source_file:

            reader = csv.DictReader(source_file)

            with temporary_train_path.open(
                "x",
                encoding="utf-8",
                newline="",
            ) as train_file, temporary_validation_path.open(
                "x",
                encoding="utf-8",
                newline="",
            ) as validation_file:

                train_writer = csv.DictWriter(
                    train_file,
                    fieldnames=["body", "label"],
                )

                validation_writer = csv.DictWriter(
                    validation_file,
                    fieldnames=["body", "label"],
                )

                train_writer.writeheader()
                validation_writer.writeheader()

                for row in reader:

                    body = normalize_body(row["body"])
                    label = row["label"].strip()

                    body_hash = hash_body(body)

                    clean_row = {
                        "body": body,
                        "label": label,
                    }

                    if body_hash in validation_groups:
                        validation_writer.writerow(clean_row)

                        validation_rows += 1
                        validation_label_counts[label] += 1

                    elif body_hash in model_train_groups:
                        train_writer.writerow(clean_row)

                        train_rows += 1
                        train_label_counts[label] += 1

                    else:
                        raise RuntimeError(
                            "An email group was not assigned "
                            "to either partition."
                        )

        # Publish the completed files only after both writes succeed.
        temporary_train_path.replace(NEW_TRAIN_PATH)
        temporary_validation_path.replace(VALIDATION_PATH)

    finally:
        if temporary_train_path.exists():
            temporary_train_path.unlink()

        if temporary_validation_path.exists():
            temporary_validation_path.unlink()

    # --------------------------------------------------
    # Pass 4: Verify output files and group separation.
    # --------------------------------------------------

    print("\n[4/5] Verifying the new datasets...")

    train_hashes = set()
    validation_hashes = set()

    def collect_hashes(file_path, destination):
        with file_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:
                destination.add(
                    hash_body(row["body"])
                )

    collect_hashes(
        NEW_TRAIN_PATH,
        train_hashes,
    )

    collect_hashes(
        VALIDATION_PATH,
        validation_hashes,
    )

    overlap = train_hashes.intersection(
        validation_hashes
    )

    if overlap:
        raise RuntimeError(
            f"Verification failed: {len(overlap):,} "
            "text groups occur in both new datasets."
        )

    if train_rows + validation_rows != total_rows:
        raise RuntimeError(
            "Verification failed: output row counts "
            "do not match the source row count."
        )

    # --------------------------------------------------
    # Pass 5: Save the report.
    # --------------------------------------------------

    print("\n[5/5] Saving the split report...")

    report = {
        "source_dataset": str(
            SOURCE_PATH.relative_to(PROJECT_ROOT)
        ),
        "training_dataset": str(
            NEW_TRAIN_PATH.relative_to(PROJECT_ROOT)
        ),
        "validation_dataset": str(
            VALIDATION_PATH.relative_to(PROJECT_ROOT)
        ),
        "random_seed": RANDOM_SEED,
        "target_validation_size": VALIDATION_SIZE,
        "source_rows": total_rows,
        "training_rows": train_rows,
        "validation_rows": validation_rows,
        "training_row_percentage": round(
            train_rows / total_rows * 100, 2
        ),
        "validation_row_percentage": round(
            validation_rows / total_rows * 100, 2
        ),
        "training_unique_text_groups": len(train_hashes),
        "validation_unique_text_groups": len(
            validation_hashes
        ),
        "overlapping_text_groups": len(overlap),
        "conflicting_label_groups": conflicting_groups,
        "training_label_counts": dict(
            sorted(train_label_counts.items())
        ),
        "validation_label_counts": dict(
            sorted(validation_label_counts.items())
        ),
        "original_train_dataset_modified": False,
        "original_test_dataset_modified": False,
    }

    with REPORT_PATH.open(
        "x",
        encoding="utf-8",
    ) as report_file:

        json.dump(
            report,
            report_file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n--- Validation Split Results ---")
    print(f"Source rows: {total_rows:,}")
    print(f"New training rows: {train_rows:,}")
    print(f"Validation rows: {validation_rows:,}")
    print(
        f"Training unique texts: {len(train_hashes):,}"
    )
    print(
        "Validation unique texts: "
        f"{len(validation_hashes):,}"
    )
    print(f"Overlapping text groups: {len(overlap)}")

    print("\nTraining label distribution:")
    for label, count in sorted(train_label_counts.items()):
        print(f"  Label {label}: {count:,}")

    print("\nValidation label distribution:")
    for label, count in sorted(
        validation_label_counts.items()
    ):
        print(f"  Label {label}: {count:,}")

    print(f"\nReport saved to: {REPORT_PATH}")
    print("\nValidation split completed successfully.")


if __name__ == "__main__":
    create_validation_split()
