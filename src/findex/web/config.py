import functools
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    index_path: Path = Field(
        default=Path("index.bin"),
        validation_alias=AliasChoices("INDEX_PATH", "FINDEX_INDEX_PATH", "index_path"),
    )
    host: str = Field(
        default="0.0.0.0",
        validation_alias=AliasChoices("HOST", "FINDEX_HOST", "host"),
    )
    port: int = Field(
        default=8000,
        validation_alias=AliasChoices("PORT", "FINDEX_PORT", "port"),
    )
    workers: int = Field(
        default=1,
        validation_alias=AliasChoices("WORKERS", "FINDEX_WORKERS", "workers"),
    )


@functools.lru_cache
def get_settings() -> Settings:
    return Settings()
