"""Settings loaded from environment variables or a ``.env`` file."""

from functools import lru_cache
from typing import Self

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
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
    groq_grade_model: str = "openai/gpt-oss-20b"  # grades the excerpts and rewrites the query; has its own limits
    groq_tokens_per_minute: int = Field(default=8000, gt=0)

    # Fetching. A PDF file is accepted up to pdf_max_bytes; 0 switches PDF support off.
    fetch_timeout_seconds: float = Field(default=20.0, gt=0)
    fetch_max_bytes: int = Field(default=5_000_000, gt=0)
    pdf_max_bytes: int = Field(default=10_000_000, ge=0)
    pdf_max_pages: int = Field(default=100, gt=0)
    pdf_max_seconds: float = Field(default=30.0, gt=0)  # reading stops after this long, which matters on a slow host

    # A stored copy of a source older than this is loaded again. 0 never loads a copy again because of its age.
    source_max_age_hours: int = Field(default=0, ge=0)
    # The website is asked whether a stored page changed before the copy is used, and the copy is made again if it did.
    # A page that was checked less than this many seconds ago is not asked about again; 0 asks at every question.
    source_check_seconds: int = Field(default=0, ge=0)

    # The web server. It answers only to this computer, unless it is hosted: then web_allowed_hosts lists the names it
    # answers to, separated by commas (a hosting service gives its own, such as name.onrender.com). When more than
    # web_max_stored_chunks chunks are stored, the stored sources are deleted before the next page is loaded; 0 keeps
    # them all. Each visitor (told apart by network address) is limited, so that nobody can use up the free plans.
    web_allowed_hosts: str = ""
    web_max_stored_chunks: int = Field(default=0, ge=0)
    rate_limit_enabled: bool = True
    rate_limit_asks_per_minute: int = Field(default=6, gt=0)
    rate_limit_asks_per_day: int = Field(default=60, gt=0)
    rate_limit_loads_per_hour: int = Field(default=20, gt=0)

    # Limits for one question
    agent_max_steps: int = Field(default=24, gt=0)  # the most graph nodes run for one question
    agent_max_link_hops: int = Field(default=3, ge=0)  # the most times links are chosen; 0 keeps to the given pages
    agent_max_pages: int = Field(
        default=5, ge=0
    )  # the most pages loaded by following links; 0 keeps to the given pages
    agent_max_llm_calls: int = Field(default=12, gt=0)
    agent_max_rewrites: int = Field(default=2, ge=0)
    # How many times the excerpts are widened with the text next to the best ones when they are not enough. It is off
    # (0) until it has been measured; each expansion needs 2 more steps and 1 more call than the limits above allow.
    agent_max_expansions: int = Field(default=0, ge=0)
    # A file that gets one JSON line for every question: the question, the steps taken and whether the agent answered.
    # It stays on this machine and is off while blank.
    agent_trace_file: str = ""

    # Limits for one evaluation run
    eval_max_cohere_calls: int = Field(default=300, gt=0)
    eval_max_groq_calls: int = Field(default=200, gt=0)
    eval_max_groq_tokens: int = Field(default=120_000, gt=0)
    eval_judge_model: str = "openai/gpt-oss-20b"
    # The baseline that crawls a site before answering: how many pages it loads, and how many links deep it goes.
    eval_crawl_max_pages: int = Field(default=40, gt=0)
    eval_crawl_max_depth: int = Field(default=2, gt=0)

    @property
    def groq_keys(self) -> tuple[SecretStr, ...]:
        """The Groq keys in the order they are used: the main key, then the backups, without blanks or repeats."""
        candidates = [self.groq_api_key.get_secret_value(), *self.groq_backup_api_keys.get_secret_value().split(",")]
        keys: list[str] = []
        for candidate in (item.strip() for item in candidates):
            if candidate and candidate not in keys:
                keys.append(candidate)
        return tuple(SecretStr(key) for key in keys)

    @property
    def web_host_names(self) -> frozenset[str]:
        """The names the web server answers to besides this computer, in lower case, without blanks or ports."""
        names = (item.strip().lower() for item in self.web_allowed_hosts.split(","))
        return frozenset(name.rsplit(":", 1)[0] if name.count(":") == 1 else name for name in names if name)

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


def describe_validation_error(error: ValidationError) -> str:
    """Name each setting that is missing or invalid and say why, without showing any value.

    Args:
        error: The error raised when the settings were loaded.

    Returns:
        One text with a part for each setting. A key must never reach the output, so values are left out.
    """
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc']).upper() or 'SETTINGS'}: {item['msg']}" for item in error.errors()
    )


@lru_cache
def get_settings() -> Settings:
    """Return the settings, loading them once.

    Returns:
        The cached settings instance.

    Raises:
        pydantic.ValidationError: If a required variable is missing or a value is invalid.
    """
    return Settings()  # type: ignore[call-arg]  # required values are read from the environment
