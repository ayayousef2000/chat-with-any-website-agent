"""Settings loaded from environment variables or a ``.env`` file."""

from functools import lru_cache
from typing import Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The output sizes that Cohere's Embed v5 models support.
EMBED_DIMENSIONS = (256, 512, 768, 1024, 1536, 2048)

_SECRET_FIELDS = ("cohere_api_key", "weaviate_api_key", "groq_api_key")


class Settings(BaseSettings):
    """Settings read from environment variables or a ``.env`` file.

    Every variable in ``.env.example`` is a field here, and a test keeps the two in step. Keys are ``SecretStr``
    values, so they do not appear in logs, error messages or ``repr`` output.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Embeddings and reranking (Cohere)
    cohere_api_key: SecretStr
    cohere_embed_model: str = "embed-v5.0-pro"
    cohere_embed_dimension: int = 1024
    cohere_rerank_model: str = "rerank-v4.0-fast"
    cohere_rerank_requests_per_minute: int = Field(default=10, gt=0)

    # Vector database (Weaviate Cloud). One collection holds a tenant for each environment.
    weaviate_url: str
    weaviate_api_key: SecretStr
    weaviate_collection: str = "WebsiteChunk"
    weaviate_tenant: str = Field(default="development", pattern=r"^[A-Za-z0-9_-]{4,64}$")
    weaviate_init_timeout_seconds: int = Field(default=30, gt=0)

    # LLM (Groq). The backup keys are separated by commas and used in order after the main key.
    groq_api_key: SecretStr
    groq_backup_api_keys: SecretStr = SecretStr("")
    groq_model: str = "openai/gpt-oss-120b"
    groq_tokens_per_minute: int = Field(default=8000, gt=0)

    # Fetching. A PDF file is accepted up to pdf_max_bytes; 0 switches PDF support off.
    fetch_timeout_seconds: float = Field(default=20.0, gt=0)
    fetch_max_bytes: int = Field(default=5_000_000, gt=0)
    pdf_max_bytes: int = Field(default=10_000_000, ge=0)
    pdf_max_pages: int = Field(default=100, gt=0)
    pdf_max_seconds: float = Field(default=30.0, gt=0)  # reading stops after this long, which matters on a slow host

    # Limits for one question
    agent_max_steps: int = Field(default=12, gt=0)  # the most graph nodes run for one question
    agent_max_link_hops: int = Field(default=3, ge=0)
    agent_max_llm_calls: int = Field(default=8, gt=0)
    agent_max_rewrites: int = Field(default=2, ge=0)

    # Limits for one evaluation run
    eval_max_cohere_calls: int = Field(default=300, gt=0)
    eval_max_groq_calls: int = Field(default=200, gt=0)
    eval_max_groq_tokens: int = Field(default=120_000, gt=0)
    eval_judge_model: str = "openai/gpt-oss-20b"

    @property
    def groq_keys(self) -> tuple[SecretStr, ...]:
        """The Groq keys in the order they are used: the main key, then the backups, without blanks or repeats."""
        candidates = [self.groq_api_key.get_secret_value(), *self.groq_backup_api_keys.get_secret_value().split(",")]
        keys: list[str] = []
        for candidate in (item.strip() for item in candidates):
            if candidate and candidate not in keys:
                keys.append(candidate)
        return tuple(SecretStr(key) for key in keys)

    @field_validator("cohere_embed_dimension")
    @classmethod
    def _check_embed_dimension(cls, value: int) -> int:
        if value not in EMBED_DIMENSIONS:
            raise ValueError(f"must be one of {EMBED_DIMENSIONS}")
        return value

    @field_validator("weaviate_url")
    @classmethod
    def _check_weaviate_url(cls, value: str) -> str:
        cleaned = value.strip().rstrip("/")
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    @model_validator(mode="after")
    def _check_secrets_are_not_blank(self) -> Self:
        for name in _SECRET_FIELDS:
            if not getattr(self, name).get_secret_value().strip():
                raise ValueError(f"{name.upper()} must not be empty")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the settings, loading them once.

    Returns:
        The cached settings instance.

    Raises:
        pydantic.ValidationError: If a required variable is missing or a value is invalid.
    """
    return Settings()  # type: ignore[call-arg]  # required values are read from the environment
