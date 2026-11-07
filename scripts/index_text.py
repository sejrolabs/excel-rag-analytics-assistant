import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import settings
from src.retrieval.indexer import index_transactions


def main():
    print("=" * 70)
    print("RECONSTRUCTION DE L'INDEX VECTORIEL")
    print("=" * 70)

    def on_progress(indexed: int, total: int):
        print(f"Progression : {indexed:,}/{total:,} ({indexed / total:.1%})")

    print(f"\nIndexation depuis la base : {settings.duckdb_path}")
    count = index_transactions(progress_callback=on_progress)

    print("\n" + "=" * 70)
    print(f"Indexation terminée avec succès : {count:,} vecteurs dans Qdrant.")
    print("=" * 70)


if __name__ == "__main__":
    main()
