from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_provider: str = "ollama"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    embedding_model: str = "intfloat/multilingual-e5-small"
    duckdb_path: str = "data/processed/transactions.duckdb"
    qdrant_path: str = "data/processed/qdrant"
    qdrant_collection: str = "transaction_descriptions"
    max_sql_rows: int = 1000
    sql_timeout_seconds: int = 10

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def ensure_directories(self) -> None:
        Path(self.duckdb_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.qdrant_path).parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
