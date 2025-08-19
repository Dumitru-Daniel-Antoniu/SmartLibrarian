import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    OPENAI_API_KEY: str
    EMBED_MODEL: str
    CHAT_MODEL: str
    CHROMA_PATH: str
    DATASET_PATH: str
    COLLECTION_NAME: str
    TOP_K: int
    TEMPERATURE: float


def _get_env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and (val is None or val == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise RuntimeError(f"Invalid int for {name}: {raw}") from e


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise RuntimeError(f"Invalid float for {name}: {raw}") from e


def _load_settings() -> Settings:
    api_key = _get_env("OPENAI_API_KEY", required=True)

    embed_model = _get_env("EMBED_MODEL", "text-embedding-3-small")
    chat_model = _get_env("CHAT_MODEL", "gpt-4o-mini")
    chroma_path = _get_env("CHROMA_PATH", "storage/chroma_db")
    collection_name = _get_env("COLLECTION_NAME", "books")
    top_k = _get_int("TOP_K", 4)
    temperature = _get_float("TEMPERATURE", 0.4)


    return Settings(
        OPENAI_API_KEY=api_key,
        EMBED_MODEL=embed_model,
        CHAT_MODEL=chat_model,
        CHROMA_PATH=chroma_path,
        COLLECTION_NAME=collection_name,
        TOP_K=top_k,
        TEMPERATURE=temperature
    )


settings = _load_settings()


def require_openai_key() -> None:
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured.")


# Verification step to ensure the variables are loaded correctly
def print_config_summary() -> None:
    print(
        "Config:\n"
        f"  EMBED_MODEL    = {settings.EMBED_MODEL}\n"
        f"  CHAT_MODEL     = {settings.CHAT_MODEL}\n"
        f"  CHROMA_PATH    = {settings.CHROMA_PATH}\n"
        f"  COLLECTION     = {settings.COLLECTION_NAME}\n"
        f"  TOP_K          = {settings.TOP_K}\n"
        f"  TEMPERATURE    = {settings.TEMPERATURE}\n"
    )
