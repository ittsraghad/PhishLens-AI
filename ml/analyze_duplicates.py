import csv
import sys
import sqlite3
import hashlib
import json
import tempfile

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


# Allow CSV cells larger than Python's default field-size limit.
csv.field_size_limit(sys.maxsize)

# Locate project paths.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT / "data" / "raw" / "Merged_Dataset.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"

REPORT_PATH = REPORT_DIR / "duplicate_analysis_report.json"


def analyze_duplicates():
    # Make sure the report directory exists.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    empty_bodies = 0
    empty_labels = 0
    non_empty_bodies = 0

    label_counts = Counter()

    print("Starting duplicate analysis...")
    print(f"Dataset: {DATASET_PATH}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    # Create a temporary SQLite database on disk.
    # It will be deleted automatically after analysis.
    with tempfile.TemporaryDirectory(
        prefix="phishlens_duplicates_",
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
                    body_hash,
                    label,
                    occurrences
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

                    body_is_empty = (
                        body is None or not body.strip()
                    )

                    label_is_empty = (
                        label is None or not label.strip()
                    )

                    if body_is_empty:
                        empty_bodies += 1

                    if label_is_empty:
                        empty_labels += 1

                    if not label_is_empty:
                        label = label.strip()
                        label_counts[label] += 1
                    else:
                        # An empty string represents a missing label.
                        label = ""

                    # Duplicate analysis requires non-empty email text.
                    if body_is_empty:
                        continue

                    non_empty_bodies += 1

                    # Match the previous analysis:
                    # remove leading/trailing whitespace only.
                    normalized_body = body.strip()

                    # Store a hash instead of the actual email text.
                    body_hash = hashlib.sha256(
                        normalized_body.encode("utf-8")
                    ).hexdigest()

                    batch.append((body_hash, label))

                    # Process batches to avoid excessive memory use.
                    if len(batch) >= batch_size:
                        connection.executemany(
                            insert_query,
                            batch,
                        )
                        connection.commit()
                        batch.clear()

                # Process any remaining records.
                if batch:
                    connection.executemany(
                        insert_query,
                        batch,
                    )
                    connection.commit()

            # Count unique non-empty email bodies.
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

            # Count distinct email texts that appear more than once.
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

            # Count repeated occurrences beyond the first copy.
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

            # Find duplicate email texts assigned multiple known labels.
            conflicting_label_groups = connection.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT body_hash
                    FROM body_labels
                    WHERE label != ''
                    GROUP BY body_hash
                    HAVING COUNT(DISTINCT label) > 1
                )
                """
            ).fetchone()[0]

            # Find duplicate groups where at least one label is missing.
            duplicate_groups_with_missing_label = connection.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT
                        body_hash,
                        SUM(occurrences) AS total_count,
                        SUM(
                            CASE
                                WHEN label = '' THEN occurrences
                                ELSE 0
                            END
                        ) AS missing_label_count
                    FROM body_labels
                    GROUP BY body_hash
                    HAVING
                        SUM(occurrences) > 1
                        AND SUM(
                            CASE
                                WHEN label = '' THEN occurrences
                                ELSE 0
                            END
                        ) > 0
                )
                """
            ).fetchone()[0]

    # Build a report after the temporary database is removed.
    report = {
        "analysis_timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": DATASET_PATH.name,
        "total_rows": total_rows,
        "non_empty_email_bodies": non_empty_bodies,
        "empty_email_bodies": empty_bodies,
        "empty_labels": empty_labels,
        "label_distribution": dict(sorted(label_counts.items())),
        "unique_non_empty_email_bodies": unique_bodies,
        "duplicate_body_groups": duplicate_groups,
        "repeated_occurrences_beyond_first_copy": (
            repeated_occurrences
        ),
        "duplicate_groups_with_conflicting_labels": (
            conflicting_label_groups
        ),
        "duplicate_groups_with_missing_labels": (
            duplicate_groups_with_missing_label
        ),
        "notes": [
            "Email bodies are compared after stripping "
            "leading and trailing whitespace.",
            "Email text is hashed; the report does not contain "
            "the original email bodies.",
            "Missing labels are excluded when checking for "
            "conflicting known labels.",
            "No original dataset rows are modified or deleted."
        ]
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

    # Display the main findings.
    print("\n--- Duplicate Analysis Results ---")
    print(f"Total dataset rows: {total_rows:,}")
    print(f"Non-empty email bodies: {non_empty_bodies:,}")
    print(f"Unique email bodies: {unique_bodies:,}")
    print(f"Duplicate body groups: {duplicate_groups:,}")
    print(
        "Repeated occurrences beyond the first copy: "
        f"{repeated_occurrences:,}"
    )
    print(
        "Duplicate groups with conflicting labels: "
        f"{conflicting_label_groups:,}"
    )
    print(
        "Duplicate groups with missing labels: "
        f"{duplicate_groups_with_missing_label:,}"
    )

    print(f"\nReport saved to: {REPORT_PATH}")
    print("\nAnalysis completed. No original data was modified.")


if __name__ == "__main__":
    analyze_duplicates()