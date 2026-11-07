from dataclasses import dataclass, asdict
import polars as pl


RECOMMENDED_COLUMNS = {
    "transaction_id", "customer_id", "transaction_date", "merchant",
    "category", "amount", "transaction_type", "city", "country",
    "payment_method", "description"
}


@dataclass
class ValidationReport:
    row_count: int
    column_count: int
    missing_recommended_columns: list[str]
    duplicate_transaction_ids: int
    null_counts: dict[str, int]
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def validate_dataframe(df: pl.DataFrame) -> ValidationReport:
    if df.is_empty():
        raise ValueError("The uploaded dataset is empty.")

    columns = set(df.columns)
    missing = sorted(RECOMMENDED_COLUMNS - columns)
    warnings = []
    if "transaction_id" not in columns:
        warnings.append("No transaction_id column: duplicate-ID check skipped.")
    if "amount" in columns and not df.schema["amount"].is_numeric():
        warnings.append("The amount column is not numeric; convert it before SQL analysis.")

    duplicate_ids = 0
    if "transaction_id" in columns:
        duplicate_ids = int(df.select(pl.col("transaction_id").is_duplicated().sum()).item())

    null_counts = {
        col: int(df.select(pl.col(col).null_count()).item())
        for col in df.columns
    }

    return ValidationReport(
        row_count=df.height,
        column_count=df.width,
        missing_recommended_columns=missing,
        duplicate_transaction_ids=duplicate_ids,
        null_counts=null_counts,
        warnings=warnings,
    )
