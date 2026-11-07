import polars as pl
from src.processing.data_processor import clean_dataframe, normalize_column_name


def test_normalize_column_name():
    assert normalize_column_name("Transaction Date") == "transaction_date"


def test_clean_dataframe_converts_amount():
    df = pl.DataFrame({"Amount": ["12,50", "20,00"]})
    cleaned = clean_dataframe(df)
    assert cleaned["amount"].to_list() == [12.5, 20.0]
