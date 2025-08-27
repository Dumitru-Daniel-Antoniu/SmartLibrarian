import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv


"""
Configuration loader for SmartLibrarian.
Loads environment variables, validates settings, and provides access to application configuration for models, vectorstore, and runtime parameters.
"""


load_dotenv()


@dataclass(frozen=True)
class Settings:
    OPENAI_API_KEY: str
    EMBED_MODEL: str
    CHAT_MODEL: str
    CHROMA_PATH: str
    COLLECTION_NAME: str
    TOP_K: int
    TEMPERATURE: float
    MAX_DISTANCE: float
    MIN_RESULTS: int


def _get_env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    """
    Retrieve an environment variable value, with optional default and required check.

    Args:
        name (str): Name of the environment variable.
        default (Optional[str]): Default value if not set.
        required (bool): Whether the variable is required.

    Returns:
        str: The environment variable value.

    Raises:
        RuntimeError: If required and not set.
    """
    val = os.getenv(name, default)
    if required and (val is None or val == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def _get_int(name: str, default: int) -> int:
    """
    Retrieve an environment variable and convert it to an integer.

    Args:
        name (str): Name of the environment variable.
        default (int): Default value if not set.

    Returns:
        int: The integer value.

    Raises:
        RuntimeError: If conversion fails.
    """
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise RuntimeError(f"Invalid int for {name}: {raw}") from e


def _get_float(name: str, default: float) -> float:
    """
    Retrieve an environment variable and convert it to a float.

    Args:
        name (str): Name of the environment variable.
        default (float): Default value if not set.

    Returns:
        float: The float value.

    Raises:
        RuntimeError: If conversion fails.
    """
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise RuntimeError(f"Invalid float for {name}: {raw}") from e


def _load_settings() -> Settings:
    """
    Load all configuration settings from environment variables.

    Returns:
        Settings: Dataclass containing all configuration values.

    Raises:
        RuntimeError: If required variables are missing or invalid.
    """
    api_key = _get_env("OPENAI_API_KEY", required=True)

    embed_model = _get_env("EMBED_MODEL", "text-embedding-3-small")
    chat_model = _get_env("CHAT_MODEL", "gpt-4o-mini")
    chroma_path = _get_env("CHROMA_PATH", "storage/chroma_db")
    collection_name = _get_env("COLLECTION_NAME", "books")
    top_k = _get_int("TOP_K", 4)
    temperature = _get_float("TEMPERATURE", 0.4)
    max_distance = _get_float("MAX_DISTANCE", 0.65)
    min_results = _get_int("MIN_RESULTS", 1)

    return Settings(
        OPENAI_API_KEY=api_key,
        EMBED_MODEL=embed_model,
        CHAT_MODEL=chat_model,
        CHROMA_PATH=chroma_path,
        COLLECTION_NAME=collection_name,
        TOP_K=top_k,
        TEMPERATURE=temperature,
        MAX_DISTANCE=max_distance,
        MIN_RESULTS=min_results
    )


settings = _load_settings()


def require_openai_key() -> None:
    """
    Ensure the OpenAI API key is configured.

    Raises:
        RuntimeError: If the API key is missing.
    """
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured.")


# Verification step to ensure the variables are loaded correctly
def print_config_summary() -> None:
    """
    Print a summary of the current configuration settings to the console.
    """
    print(
        "Config:\n"
        f"  EMBED_MODEL    = {settings.EMBED_MODEL}\n"
        f"  CHAT_MODEL     = {settings.CHAT_MODEL}\n"
        f"  CHROMA_PATH    = {settings.CHROMA_PATH}\n"
        f"  COLLECTION     = {settings.COLLECTION_NAME}\n"
        f"  TOP_K          = {settings.TOP_K}\n"
        f"  TEMPERATURE    = {settings.TEMPERATURE}\n"
        f"  MAX_DISTANCE   = {settings.MAX_DISTANCE}\n"
        f"  MIN_RESULTS    = {settings.MIN_RESULTS}\n"
    )
