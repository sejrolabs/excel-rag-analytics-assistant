import re
import polars as pl


def normalize_column_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def _clean_amount_expr(col_name: str) -> pl.Expr:
    """Nettoie une colonne de montant contenant des devises et divers formats numériques."""
    s = (
        pl.col(col_name)
        .cast(pl.Utf8)
        .str.replace_all(r"[\s\u00a0\u202f€$£]", "")
        .str.replace_all(r"[A-Za-z]+", "")
    )
    return (
        pl.when(s.str.contains(r",\d{3}\."))
        .then(s.str.replace_all(",", ""))
        .when(s.str.contains(r"\.\d{3},"))
        .then(s.str.replace_all(r"\.", "").str.replace(",", "."))
        .otherwise(s.str.replace(",", "."))
        .cast(pl.Float64, strict=False)
    )


def clean_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize column names and perform robust type conversions."""
    # 1. Normalisation des noms de colonnes
    col_mapping = {c: normalize_column_name(c) for c in df.columns}
    normalized_names = list(col_mapping.values())

    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError("Duplicate column names detected after normalization.")

    renamed = df.rename(col_mapping)

    # 2. Nettoyage et conversion robuste des dates
    if "transaction_date" in renamed.columns:
        col_type = renamed.schema["transaction_date"]
        if col_type in (pl.Datetime, pl.Date):
            renamed = renamed.with_columns(
                pl.col("transaction_date").cast(pl.Date, strict=False)
            )
        else:
            # En Polars 2.0+, le cast direct string -> date est interdit, utiliser str.to_date()
            date_expr = pl.coalesce([
                pl.col("transaction_date").str.to_date("%d/%m/%Y", strict=False),
                pl.col("transaction_date").str.to_date("%Y-%m-%d", strict=False),
                pl.col("transaction_date").str.to_date("%d-%m-%Y", strict=False),
                pl.col("transaction_date").str.to_date(strict=False),
            ])
            renamed = renamed.with_columns(date_expr.alias("transaction_date"))

    # 3. Nettoyage robuste des montants (espaces, devises, séparateurs décimaux)
    if "amount" in renamed.columns:
        renamed = renamed.with_columns(_clean_amount_expr("amount").alias("amount"))

    # 4. Nettoyage des espaces de début/fin sur toutes les colonnes textuelles
    for col in renamed.columns:
        if renamed.schema[col] in (pl.String, pl.Utf8):
            renamed = renamed.with_columns(pl.col(col).str.strip_chars())

    return renamed


def dataframe_to_records(df: pl.DataFrame, text_column: str = "description") -> list[dict]:
    if text_column not in df.columns:
        return []
    useful = [
        c for c in [
            "transaction_id", "customer_id", "transaction_date",
            "category", "city", "amount", text_column
        ] if c in df.columns
    ]
    return df.select(useful).drop_nulls([text_column]).to_dicts()
