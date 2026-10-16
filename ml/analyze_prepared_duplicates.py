import csv
import sys
import sqlite3
import hashlib
import json
import tempfile

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


csv.field_size_limit(sys.maxsize)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "prepared_emails.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"

REPORT_PATH = REPORT_DIR / "prepared_duplicates_report.json"


def analyze_prepared_duplicates():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    label_counts = Counter()

    print("Starting prepared dataset duplicate analysis...")
    print(f"Dataset: {DATASET_PATH}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {DATASET_PATH}"
        )

    # Use a temporary on-disk database to limit memory usage.
    with tempfile.TemporaryDirectory(
        prefix="phishlens_prepared_duplicates_",
        dir=REPORT_DIR,
    ) as temporary_directory:

        database_path = (
            Path(temporary_directory) / "duplicates.sqlite3"
        )

        with sqlite3.connect(database_path) as connection:

            connection.execute(
                """
                CREATE TABLE body_labels (
                    body_hash TEXT NOT NULL,
                    label TEXT NOT NULL,
                    occurrences INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (body_hash, label)
                )
                """
            )

            insert_query = """
                INSERT INTO body_labels (
                    body_hash, label, occurrences
                )
                VALUES (?, ?, 1)
                ON CONFLICT(body_hash, label)
                DO UPDATE SET
                    occurrences = occurrences + 1
            """

            batch = []
            batch_size = 5000

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
                        continue

                    if label is None or not label.strip():
                        continue

                    body = body.strip()
                    label = label.strip()

                    label_counts[label] += 1

                    body_hash = hashlib.sha256(
                        body.encode("utf-8")
                    ).hexdigest()

                    batch.append((body_hash, label))

                    if len(batch) >= batch_size:
                        connection.executemany(
                            insert_query,
                            batch,
                        )
                        connection.commit()
                        batch.clear()

                if batch:
                    connection.executemany(
                        insert_query,
                        batch,
                    )
                    connection.commit()

            # Count unique email bodies.
            unique_bodies = connection.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT body_hash
                    FROM body_labels
                    GROUP BY body_hash
                )
                """
            ).fetchone()[0]

            # Count body groups appearing more than once.
            duplicate_groups = connection.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT body_hash
                    FROM body_labels
                    GROUP BY body_hash
                    HAVING SUM(occurrences) > 1
                )
                """
            ).fetchone()[0]

            # Count occurrences beyond the first copy.
            repeated_occurrences = connection.execute(
                """
                SELECT COALESCE(SUM(total_count - 1), 0)
                FROM (
                    SELECT SUM(occurrences) AS total_count
                    FROM body_labels
                    GROUP BY body_hash
                    HAVING SUM(occurrences) > 1
                )
                """
            ).fetchone()[0]

            # Find duplicate texts assigned more than one label.
            conflicting_label_groups = connection.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT body_hash
                    FROM body_labels
                    GROUP BY body_hash
                    HAVING COUNT(DISTINCT label) > 1
                )
                """
            ).fetchone()[0]

    report = {
        "analysis_timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": DATASET_PATH.name,
        "total_rows_read": total_rows,
        "label_distribution": dict(sorted(label_counts.items())),
        "unique_non_empty_email_bodies": unique_bodies,
        "duplicate_body_groups": duplicate_groups,
        "repeated_occurrences_beyond_first_copy": (
            repeated_occurrences
        ),
        "duplicate_groups_with_conflicting_labels": (
            conflicting_label_groups
        ),
        "comparison_method": (
            "SHA-256 hashes of email bodies after stripping "
            "leading and trailing whitespace."
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

    print("\n--- Prepared Dataset Duplicate Results ---")
    print(f"Total rows read: {total_rows:,}")
    print(f"Unique email bodies: {unique_bodies:,}")
    print(f"Duplicate body groups: {duplicate_groups:,}")
    print(
        "Repeated occurrences beyond first copy: "
        f"{repeated_occurrences:,}"
    )
    print(
        "Duplicate groups with conflicting labels: "
        f"{conflicting_label_groups:,}"
    )

    print(f"\nReport saved to: {REPORT_PATH}")
    print("Analysis completed. No dataset was modified.")


if __name__ == "__main__":
    analyze_prepared_duplicates()