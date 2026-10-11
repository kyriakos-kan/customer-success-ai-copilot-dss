"""Central settings, read from environment variables (.env in Docker)."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore",
                                      protected_namespaces=("settings_",))

    database_url: str = "sqlite:///./copilot.db"   # local dev fallback
    chroma_host: str | None = None                  # None -> RAG runs in-memory
    chroma_port: int = 8000
    model_dir: Path = Path("./models")
    playbook_path: Path = Path("playbook/nba_playbook.yaml")

    anthropic_api_key: str | None = None
    # Writer backend: auto | template | ollama | anthropic  (see agents/writer.py)
    writer_backend: str = "auto"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "mistral:7b"        # Apache 2.0, no usage conditions
    # Only the top-N accounts by revenue at risk get an LLM draft per run; the rest use
    # the grounded template. Keeps local models (Ollama on CPU) fast. 0 = no limit.
    llm_max_drafts: int = 15
    # Seconds to wait for one Ollama draft before falling back to the template
    ollama_timeout_seconds: int = 600
    writer_model: str = "claude-sonnet-5-5"

    pipeline_interval_seconds: int = 86400
    risk_high: float = 0.65
    risk_medium: float = 0.35
    # Minimum retrieved context documents before the writer may draft
    min_context_docs: int = 1
    # Don't draft again for an account whose outreach was approved within this many days
    recontact_cooldown_days: int = 30
    # Pilot control group: this % of flagged accounts gets no draft (handled as today), so
    # retention can be compared fairly. Assignment is random but fixed per account. 0 = off.
    control_group_pct: int = 0


@lru_cache
def get_settings() -> Settings:
    return Settings()
