import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "phishlens_tfidf_logreg.joblib"
EXTERNAL_PATH = ROOT / "data" / "external" / "eval.parquet"
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
REPORT_PATH = (
    ROOT / "reports" / "model_evaluation" / "external_evaluation.json"
)


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_existing_hashes(path):
    df = pd.read_csv(path, usecols=["body"])
    texts = df["body"].map(normalize_text)
    return {text_hash(text) for text in texts if text}


def main():
    print("PhishLens AI - External Evaluation")
    print("No model training or model files will be changed.\n")

    for path in [MODEL_PATH, EXTERNAL_PATH, TRAIN_PATH, TEST_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("[1/5] Loading external data...")
    df = pd.read_parquet(EXTERNAL_PATH)

    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"External data must contain columns: {required}")

    df = df[["text", "label"]].copy()
    df["text"] = df["text"].map(normalize_text)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    df = df[
        df["text"].ne("")
        & df["label"].isin([0, 1])
    ].copy()
    df["label"] = df["label"].astype(int)

    original_valid_rows = len(df)
    print(f"Valid labeled rows: {original_valid_rows:,}")

    print("\n[2/5] Checking exact overlap with existing datasets...")
    train_hashes = load_existing_hashes(TRAIN_PATH)
    test_hashes = load_existing_hashes(TEST_PATH)
    existing_hashes = train_hashes | test_hashes

    df["text_hash"] = df["text"].map(text_hash)
    overlaps = df["text_hash"].isin(existing_hashes)
    overlap_rows = int(overlaps.sum())
    df = df.loc[~overlaps].copy()

    print(f"Rows matching train or test text: {overlap_rows:,}")
    print(f"Rows remaining: {len(df):,}")

    print("\n[3/5] Checking repeated texts and conflicting labels...")
    label_counts = df.groupby("text")["label"].nunique()
    conflicting_texts = set(label_counts[label_counts > 1].index)
    conflicting_rows = int(df["text"].isin(conflicting_texts).sum())

    df = df[~df["text"].isin(conflicting_texts)].copy()
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").copy()
    duplicate_rows_removed = before_dedup - len(df)

    if df.empty or df["label"].nunique() < 2:
        raise ValueError(
            "Not enough data from both classes remains for evaluation."
        )

    print(f"Rows removed due to conflicting labels: {conflicting_rows:,}")
    print(f"Repeated rows removed: {duplicate_rows_removed:,}")
    print(f"Final evaluation rows: {len(df):,}")
    print("Final label counts:")
    print(df["label"].value_counts().sort_index().to_string())

    print("\n[4/5] Loading the saved model and predicting...")
    model = joblib.load(MODEL_PATH)
    y_true = df["label"].to_numpy()
    y_pred = model.predict(df["text"].tolist())

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "evaluation_type": (
            "Exploratory external evaluation after exact-text overlap removal"
        ),
        "original_valid_external_rows": original_valid_rows,
        "exact_overlap_rows_removed": overlap_rows,
        "conflicting_label_rows_removed": conflicting_rows,
        "repeated_rows_removed": duplicate_rows_removed,
        "evaluation_rows": len(df),
        "label_counts": {
            str(label): int(count)
            for label, count in df["label"].value_counts().sort_index().items()
        },
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_class_1": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_class_1": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1_class_1": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "confusion_matrix_labels_0_1": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Safe-labeled (0)", "Phishing/spam-labeled (1)"],
            output_dict=True,
            zero_division=0,
        ),
        "limitations": [
            "Exact text overlap was checked after stripping outer whitespace.",
            "Near-duplicates, quoted replies, and related email threads may remain.",
            "The external source may overlap with the source datasets used for training.",
            "Label 1 may include spam or scam emails, not only phishing.",
            "Results are exploratory and do not establish real-world security performance.",
        ],
    }

    print("\n[5/5] Saving a separate report...")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision_class_1']:.4f}")
    print(f"Recall:    {metrics['recall_class_1']:.4f}")
    print(f"F1-score:  {metrics['f1_class_1']:.4f}")
    print("Confusion matrix (labels 0, 1):")
    print(metrics["confusion_matrix_labels_0_1"])
    print(f"\nReport saved to: {REPORT_PATH.relative_to(ROOT)}")
    print("Evaluation completed. The model and source datasets were not modified.")


if __name__ == "__main__":
    main()
