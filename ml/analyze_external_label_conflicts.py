
from pathlib import Path

import pandas as pd


EXTERNAL_PATH = Path("data/external/eval.parquet")
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")


def normalize_text(series):
    """Strip leading and trailing whitespace without changing source files."""
    return series.astype("string").str.strip()


def analyze_conflicts(external_df, existing_path, dataset_name):
    print(f"\n{'=' * 50}")
    print(f"{dataset_name.upper()} LABEL COMPARISON")
    print("=" * 50)

    existing_df = pd.read_csv(
        existing_path,
        usecols=["body", "label"],
        dtype={"body": "string"},
    )

    # Normalize text and labels in memory only.
    external = external_df[["text", "label"]].copy()
    external["text"] = normalize_text(external["text"])
    external["label"] = pd.to_numeric(
        external["label"], errors="coerce"
    )

    existing = existing_df[["body", "label"]].copy()
    existing["body"] = normalize_text(existing["body"])
    existing["label"] = pd.to_numeric(
        existing["label"], errors="coerce"
    )

    # Remove empty text and missing labels.
    external = external.dropna(subset=["text", "label"])
    external = external[external["text"] != ""]

    existing = existing.dropna(subset=["body", "label"])
    existing = existing[existing["body"] != ""]

    # Keep unique text-label combinations.
    external = external.drop_duplicates(
        subset=["text", "label"]
    )
    existing = existing.drop_duplicates(
        subset=["body", "label"]
    )

    # Rename columns to allow a text-based merge.
    external = external.rename(
        columns={"text": "body", "label": "external_label"}
    )
    existing = existing.rename(
        columns={"label": "existing_label"}
    )

    # Compare labels only for texts found in both datasets.
    matches = external.merge(
        existing,
        on="body",
        how="inner",
    )

    if matches.empty:
        print("No matching texts were found.")
        return

    # A text may have multiple label records, so compare every
    # matching label combination.
    matches["label_agrees"] = (
        matches["external_label"] == matches["existing_label"]
    )

    agreements = int(matches["label_agrees"].sum())
    conflicts = int((~matches["label_agrees"]).sum())

    print(f"Matching text-label pairs: {len(matches):,}")
    print(f"Pairs with matching labels: {agreements:,}")
    print(f"Pairs with conflicting labels: {conflicts:,}")

    print("\nLabel comparison table:")
    print(
        pd.crosstab(
            matches["existing_label"],
            matches["external_label"],
            rownames=["Existing label"],
            colnames=["External label"],
        ).to_string()
    )

    print(
        "\nNote: counts represent matching text-label pairs, "
        "not necessarily unique messages."
    )


def main():
    print("Analyzing label consistency across datasets...")

    external_df = pd.read_parquet(EXTERNAL_PATH)

    print(f"External dataset rows: {len(external_df):,}")

    for name, path in [
        ("Training", TRAIN_PATH),
        ("Testing", TEST_PATH),
    ]:
        if not path.exists():
            print(f"\nWARNING: Missing file: {path}")
            continue

        analyze_conflicts(
            external_df=external_df,
            existing_path=path,
            dataset_name=name,
        )

    print("\nLabel consistency analysis completed.")
    print("No source files were modified.")


if __name__ == "__main__":
    main()
