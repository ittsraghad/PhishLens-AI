
from pathlib import Path

import pandas as pd


# Paths to the external dataset and existing datasets.
EXTERNAL_PATH = Path("data/external/eval.parquet")
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")


def analyze_dataset():
    print("Analyzing external evaluation dataset...\n")

    # Read the external Parquet dataset.
    df = pd.read_parquet(EXTERNAL_PATH)

    # 1. Dataset dimensions.
    print("1. Dataset dimensions")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # 2. Column names and data types.
    print("\n2. Column names and data types")
    print(df.dtypes.to_string())

    # 3. Missing values.
    print("\n3. Missing values per column")
    missing = df.isna().sum()

    if missing.any():
        print(missing[missing > 0].to_string())
    else:
        print("No missing values.")

    # 4. Label distribution.
    print("\n4. Label distribution")

    if "label" in df.columns:
        label_counts = df["label"].value_counts(
            dropna=False
        ).sort_index()

        print(label_counts.to_string())

        print("\nLabel percentages:")
        label_percentages = (
            df["label"]
            .value_counts(normalize=True, dropna=False)
            .mul(100)
            .round(2)
            .sort_index()
        )

        print(label_percentages.to_string())
    else:
        print("WARNING: No 'label' column found.")

    # 5. Source distribution.
    print("\n5. Source distribution")

    if "dataset_name" in df.columns:
        print(
            df["dataset_name"]
            .value_counts(dropna=False)
            .to_string()
        )
    else:
        print("No 'dataset_name' column found.")

    # 6. Duplicate text statistics.
    print("\n6. Duplicate text statistics")

    if "text" in df.columns:
        text = df["text"].astype("string").str.strip()
        nonempty = text.notna() & text.ne("")

        nonempty_texts = text[nonempty]
        unique_texts = nonempty_texts.nunique()

        print(f"Non-empty texts: {len(nonempty_texts):,}")
        print(f"Unique non-empty texts: {unique_texts:,}")
        print(
            "Repeated occurrences: "
            f"{len(nonempty_texts) - unique_texts:,}"
        )
    else:
        print("No 'text' column found.")

    print("\nDataset analysis completed.")


def check_overlap():
    print("\n7. Checking overlap with existing datasets")

    # Load the external evaluation data.
    external_df = pd.read_parquet(EXTERNAL_PATH)

    # Normalize text by stripping leading and trailing whitespace.
    external_texts = set(
        external_df["text"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    # Exclude empty texts from the comparison.
    external_texts.discard("")

    print(
        f"Unique non-empty external texts: "
        f"{len(external_texts):,}"
    )

    # Compare against training and testing datasets.
    datasets = [
        ("Training", TRAIN_PATH),
        ("Testing", TEST_PATH),
    ]

    for name, path in datasets:
        print(f"\n{name} dataset")

        if not path.exists():
            print(f"WARNING: File not found: {path}")
            continue

        existing_df = pd.read_csv(path)

        if "body" not in existing_df.columns:
            print("WARNING: The 'body' column was not found.")
            continue

        existing_texts = set(
            existing_df["body"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        # Exclude empty texts.
        existing_texts.discard("")

        # Find exact text matches after stripping outer whitespace.
        overlap = external_texts.intersection(existing_texts)

        overlap_percentage = (
            len(overlap) / len(external_texts) * 100
            if external_texts
            else 0.0
        )

        print(f"Unique existing texts: {len(existing_texts):,}")
        print(f"Matching external texts: {len(overlap):,}")
        print(
            "Percentage of external texts matching: "
            f"{overlap_percentage:.2f}%"
        )

    print(
        "\nOverlap analysis completed. "
        "No dataset rows were modified."
    )


def main():
    analyze_dataset()
    check_overlap()


if __name__ == "__main__":
    main()
