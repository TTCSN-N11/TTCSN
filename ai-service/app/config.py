"""AI service configuration."""

import functools

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    AI_API_KEY: str = ""
    MODEL_PATH: str = "app/models/best_random_forest_model.pkl"
    MODEL_VERSION: str = ""
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@functools.lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
