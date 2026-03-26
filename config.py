from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    CHAT_PROVIDER: str = "auto"
    EMBED_PROVIDER: str = "gemini"
    SUPABASE_URL: str
    SUPABASE_KEY: str
    SMTP_EMAIL: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: Optional[str] = None

    LEGAL_DISCLAIMER: str = (
        "⚠️ This tool is for informational purposes only. It is not legal advice. "
        "Consult qualified legal counsel before making decisions."
    )

    @property
    def has_anthropic_key(self) -> bool:
        return self._is_real_key(self.ANTHROPIC_API_KEY)

    @property
    def has_openai_key(self) -> bool:
        return self._is_real_key(self.OPENAI_API_KEY)

    @property
    def has_google_key(self) -> bool:
        return self._is_real_key(self.GOOGLE_API_KEY)

    @staticmethod
    def _is_real_key(value: Optional[str]) -> bool:
        v = (value or "").strip()
        if not v:
            return False
        return v.lower() not in {"your_key_here", "changeme", "replace_me"}

    @property
    def chat_provider(self) -> str:
        p = (self.CHAT_PROVIDER or "auto").strip().lower()
        return p if p in {"auto", "anthropic", "openai", "gemini"} else "auto"

    @property
    def embed_provider(self) -> str:
        p = (self.EMBED_PROVIDER or "gemini").strip().lower()
        return p if p in {"gemini", "openai"} else "gemini"

    @property
    def has_smtp(self) -> bool:
        return bool(
            self.SMTP_EMAIL
            and self.SMTP_PASSWORD
            and self.SMTP_HOST
            and self.SMTP_PORT
        )


settings = Settings()

# RAG pipeline constants — used by core/rag/vector_store.py and core/rag/qa_system.py
CHUNK_SIZE: int = 800          # characters per text chunk when indexing
CHUNK_OVERLAP: int = 200       # overlap between consecutive chunks (sliding window)
MAX_CONTEXT_CHARS: int = 4000  # max total characters passed to LLM as context (~1000 tokens)
EMBEDDING_DIMS: int = 1536     # vector dimensions (matches regulation_embeddings table)

# Canonical, stable import path used across the app.
LEGAL_DISCLAIMER: str = settings.LEGAL_DISCLAIMER

