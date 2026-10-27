from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project settings. Values can be overridden in the .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # The chunker counts tokens with this model and the storage embeds with it,
    # so both steps must use the same one.
    embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_max_tokens: int = 256

    chunks_path: Path = Path("data/processed/chunks.jsonl")
    vector_db_path: Path = Path("data/processed/vectors.db")

    # Google AI Studio (Gemini). The key lives in .env, which is never committed.
    google_api_key: str = ""
    llm_model: str = "gemini-3.5-flash-lite"
    # Free-tier quotas are low, so calls are spaced out
    llm_requests_per_minute: int = 10

    # Number of chunks given to the LLM for each question
    retrieval_k: int = 5


settings = Settings()
