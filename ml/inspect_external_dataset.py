import json
from pathlib import Path

from huggingface_hub import HfApi


DATASET_ID = "puyang2025/seven-phishing-email-datasets"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "data_quality"
    / "external_dataset_metadata.json"
)


def main():
    print("Inspecting external dataset metadata...")
    print(f"Dataset: {DATASET_ID}")

    api = HfApi()

    try:
        info = api.dataset_info(DATASET_ID, files_metadata=True)

        files = []
        for sibling in info.siblings or []:
            files.append(
                {
                    "filename": sibling.rfilename,
                    "size_bytes": getattr(sibling, "size", None),
                }
            )

        report = {
            "dataset_id": DATASET_ID,
            "dataset_card_url": (
                f"https://huggingface.co/datasets/{DATASET_ID}"
            ),
            "revision": info.sha,
            "files": files,
            "note": (
                "This report contains repository metadata only. "
                "It does not download or inspect individual email records. "
                "Shared source datasets may overlap with the current training "
                "data, so this repository must not automatically be treated "
                "as an independent evaluation set."
            ),
        }

        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REPORT_PATH.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        print("\nFiles found:")
        for item in files:
            size = item["size_bytes"]
            size_text = (
                f"{size / (1024 * 1024):.2f} MB"
                if size is not None
                else "size unavailable"
            )
            print(f"- {item['filename']} ({size_text})")

        print(f"\nMetadata report saved to: {REPORT_PATH}")

    except Exception as error:
        print("\nCould not retrieve dataset metadata.")
        print(f"Error type: {type(error).__name__}")
        print(f"Error details: {error}")
        print(
            "\nCheck your internet connection and the dataset page. "
            "No email records were intentionally downloaded by this script."
        )


if __name__ == "__main__":
    main()