import polars as pl

FIELD_LABELS = {
    "description": "Description",
    "category": "Catégorie",
    "merchant": "Commerçant",
    "city": "Ville",
    "country": "Pays",
    "transaction_type": "Type de transaction",
    "payment_method": "Moyen de paiement",
    "amount": "Montant",
}


def enrich_transaction_text(df: pl.DataFrame) -> pl.DataFrame:
    """
    Construit un texte descriptif enrichi pour chaque transaction.
    S'adapte dynamiquement aux colonnes disponibles dans le DataFrame.
    La colonne transaction_text servira à la recherche sémantique et lexicale.
    """
    if df.is_empty():
        return df.with_columns(pl.lit("").alias("transaction_text"))

    # Identifier les champs connus présents dans le DataFrame
    available_fields = [col for col in FIELD_LABELS if col in df.columns]

    # Fallback : si aucun champ standard n'est présent, on s'appuie sur les colonnes textuelles
    if not available_fields:
        available_fields = [
            col for col in df.columns
            if df.schema[col] in (pl.String, pl.Utf8)
        ]

    if not available_fields:
        raise ValueError(
            "Aucune colonne exploitable pour générer une description textuelle des transactions."
        )

    parts: list[pl.Expr] = [pl.lit("Transaction bancaire. ")]

    for col in available_fields:
        label = FIELD_LABELS.get(col, col.replace("_", " ").capitalize())
        parts.append(pl.lit(f"{label} : "))
        if col == "amount":
            parts.append(pl.col(col).cast(pl.String).fill_null("Non renseigné"))
            parts.append(pl.lit(" EUR. "))
        else:
            parts.append(pl.col(col).cast(pl.String).fill_null("Non renseigné"))
            parts.append(pl.lit(". "))

    return df.with_columns(
        pl.concat_str(parts, separator="").str.strip_chars().alias("transaction_text")
    )
