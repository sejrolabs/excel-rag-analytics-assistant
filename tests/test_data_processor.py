from datetime import date
import polars as pl
from src.processing.data_processor import normalize_column_name, clean_dataframe


def test_normalize_column_name():
    assert normalize_column_name("Transaction ID") == "transaction_id"
    assert normalize_column_name("Montant (EUR)") == "montant_eur"
    assert normalize_column_name("Date & Heure") == "date_heure"


def test_clean_dataframe():
    df = pl.DataFrame({
        "Date de Transaction": ["2024-01-15"],
        "Montant": ["1 250,50"],
        "Description": [" Restaurant Paris "],
    })
    cleaned = clean_dataframe(df)
    assert "date_de_transaction" in cleaned.columns
    assert "montant" in cleaned.columns
    assert "description" in cleaned.columns
    assert cleaned["description"][0] == "Restaurant Paris"


def test_clean_dataframe_dates_and_currencies():
    df = pl.DataFrame({
        "Transaction Date": ["25/12/2023", "01/01/2024"],
        "Amount": [" 45,50 € ", "$1,200.00 EUR"],
    })
    cleaned = clean_dataframe(df)
    assert cleaned["transaction_date"][0] == date(2023, 12, 25)
    assert cleaned["transaction_date"][1] == date(2024, 1, 1)
    assert cleaned["amount"][0] == 45.50
    assert cleaned["amount"][1] == 1200.00
