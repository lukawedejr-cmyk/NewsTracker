from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class PipelineConfig:
    openai_api_key: str
    openai_chat_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"
    mock_openai: bool = False


def load_config(mock_openai: bool = False, env_file: str | None = None) -> PipelineConfig:
    if env_file:
        load_dotenv(env_file)
    else:
        default_env = Path(".env")
        if default_env.exists():
            load_dotenv(default_env)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip() or "text-embedding-3-small"

    if not mock_openai and not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required. Set it in a .env file or environment variable, "
            "or use --mock-openai for local/demo runs."
        )

    return PipelineConfig(
        openai_api_key=api_key,
        openai_chat_model=chat_model,
        openai_embed_model=embed_model,
        mock_openai=mock_openai,
    )
