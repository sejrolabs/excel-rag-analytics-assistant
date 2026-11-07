
import sys
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import settings
from src.ingestion.excel_loader import load_excel
from src.ingestion.data_validator import validate_dataframe
from src.processing.data_processor import clean_dataframe
from src.storage.duckdb_store import DuckDBStore


def main():
    parser = argparse.ArgumentParser(
        description="Import and prepare an Excel transaction dataset."
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the Excel file to import."
    )
    args = parser.parse_args()

    file_path = Path(args.file)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print("\n[1/4] Loading Excel file...")
    raw_df = load_excel(file_path)

    print(f"Loaded {raw_df.height:,} rows and {raw_df.width} columns.")

    print("\n[2/4] Cleaning the dataset...")
    cleaned_df = clean_dataframe(raw_df)

    print(f"Cleaned dataset: {cleaned_df.height:,} rows.")
    print("Normalized columns:", cleaned_df.columns)

    print("\n[3/4] Validating the cleaned dataset...")
    report = validate_dataframe(cleaned_df)

    print("\nValidation report:")
    for key, value in report.to_dict().items():
        print(f"- {key}: {value}")

    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings:
            print(f"  ! {warning}")

    if report.missing_recommended_columns:
        print("\nMissing recommended columns:")
        for column in report.missing_recommended_columns:
            print(f"  - {column}")

    print("\n[4/4] Saving transactions to DuckDB...")
    store = DuckDBStore(settings.duckdb_path)
    store.replace_transactions(cleaned_df)

    count = store.count_transactions()

    print("\nImport completed successfully.")
    print(f"Database: {settings.duckdb_path}")
    print(f"Transactions stored: {count:,}")


if __name__ == "__main__":
    main()
