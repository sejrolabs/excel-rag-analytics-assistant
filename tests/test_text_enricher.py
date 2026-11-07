import polars as pl
import pytest
from src.processing.text_enricher import enrich_transaction_text


def test_enrich_with_all_columns():
    df = pl.DataFrame({
        "transaction_id": [1],
        "description": ["Achat supermarche"],
        "category": ["Alimentation"],
        "merchant": ["Carrefour"],
        "city": ["Paris"],
        "country": ["France"],
        "transaction_type": ["Debit"],
        "payment_method": ["Carte"],
        "amount": [45.5],
    })
    enriched = enrich_transaction_text(df)
    assert "transaction_text" in enriched.columns
    text = enriched["transaction_text"][0]
    assert "Carrefour" in text
    assert "Alimentation" in text
    assert "45.5 EUR." in text


def test_enrich_with_missing_optional_columns():
    # Test avec colonnes partielles (doit fonctionner sans planter)
    df = pl.DataFrame({
        "transaction_id": [2],
        "description": ["Virement loyer"],
        "amount": [800.0],
    })
    enriched = enrich_transaction_text(df)
    assert "transaction_text" in enriched.columns
    text = enriched["transaction_text"][0]
    assert "Virement loyer" in text
    assert "800.0 EUR." in text


def test_enrich_empty_dataframe():
    df = pl.DataFrame()
    enriched = enrich_transaction_text(df)
    assert "transaction_text" in enriched.columns
