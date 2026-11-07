
from pathlib import Path

import polars as pl


def load_excel(
    path: str | Path,
    sheet_name: str | None = None
) -> pl.DataFrame:
    """
    Load an Excel worksheet into a Polars DataFrame.

    If sheet_name is None, the first worksheet is loaded.
    Otherwise, the specified worksheet is loaded.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    try:
        if sheet_name is None:
            return pl.read_excel(path)

        return pl.read_excel(
            path,
            sheet_name=sheet_name
        )

    except Exception as exc:
        raise ValueError(
            f"Unable to read Excel file {path}: {exc}"
        ) from exc