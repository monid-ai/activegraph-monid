"""Environment-backed configuration.

Loaded once at import time; fails fast if either API key is missing.
The monid client and the LLM provider both read their keys from here
(not from globals scattered around the codebase).
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    monid_api_key: str = Field(..., alias="MONID_API_KEY")
    anthropic_api_key: str = Field(..., alias="ANTHROPIC_API_KEY")
    monid_base_url: str = Field(
        "https://api.monid.ai", alias="MONID_BASE_URL"
    )


settings = Settings()  # raises pydantic.ValidationError if keys are missing
