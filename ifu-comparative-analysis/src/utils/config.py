"""
Configuration utilities for IFU Analysis System.

Loads configuration from environment variables and provides defaults.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from anthropic import Anthropic


def load_config() -> dict:
    """
    Load configuration from environment variables.

    Returns:
        Dictionary with configuration values
    """
    # Load .env file if it exists
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    config = {
        # Anthropic API
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),

        # Model configuration
        "model_name": os.getenv("MODEL_NAME", "claude-sonnet-4.5"),
        "temperature": float(os.getenv("MODEL_TEMPERATURE", "0.0")),
        "max_tokens": int(os.getenv("MODEL_MAX_TOKENS", "16000")),

        # Checkpoint configuration
        "checkpoint_backend": os.getenv("CHECKPOINT_BACKEND", "sqlite"),
        "checkpoint_db_path": os.getenv("CHECKPOINT_DB_PATH", "./checkpoints.db"),

        # PostgreSQL (if using)
        "postgres_host": os.getenv("POSTGRES_HOST", "localhost"),
        "postgres_port": int(os.getenv("POSTGRES_PORT", "5432")),
        "postgres_db": os.getenv("POSTGRES_DB", "ifu_analysis"),
        "postgres_user": os.getenv("POSTGRES_USER", "postgres"),
        "postgres_password": os.getenv("POSTGRES_PASSWORD", ""),

        # Analysis configuration
        "enable_human_review": os.getenv("ENABLE_HUMAN_REVIEW", "true").lower() == "true",
        "severity_threshold": os.getenv("SEVERITY_THRESHOLD", "all"),

        # Report configuration
        "output_dir": os.getenv("OUTPUT_DIR", "./reports"),
        "include_toc": os.getenv("INCLUDE_TOC", "true").lower() == "true",
        "include_summary": os.getenv("INCLUDE_SUMMARY", "true").lower() == "true",
        "include_statistics": os.getenv("INCLUDE_STATISTICS", "true").lower() == "true",

        # Language
        "document_language": os.getenv("DOCUMENT_LANGUAGE", "en"),

        # Processing
        "max_pages_to_compare": int(os.getenv("MAX_PAGES_TO_COMPARE", "100")),
        "parallel_page_processing": os.getenv("PARALLEL_PAGE_PROCESSING", "false").lower() == "true",

        # Logging
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
    }

    return config


def get_anthropic_client(api_key: Optional[str] = None) -> Anthropic:
    """
    Get configured Anthropic client.

    Args:
        api_key: Optional API key. If not provided, loads from environment.

    Returns:
        Configured Anthropic client
    """
    if not api_key:
        config = load_config()
        api_key = config["anthropic_api_key"]

    if not api_key:
        raise ValueError(
            "Anthropic API key not found. "
            "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
        )

    return Anthropic(api_key=api_key)


def validate_config(config: dict) -> tuple[bool, list[str]]:
    """
    Validate configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    # Check API key
    if not config.get("anthropic_api_key"):
        errors.append("ANTHROPIC_API_KEY is required")

    # Check model name
    valid_models = ["claude-sonnet-4.5", "claude-sonnet-4", "claude-opus-4"]
    if config.get("model_name") not in valid_models:
        errors.append(f"MODEL_NAME must be one of: {', '.join(valid_models)}")

    # Check checkpoint backend
    valid_backends = ["memory", "sqlite", "postgres"]
    if config.get("checkpoint_backend") not in valid_backends:
        errors.append(f"CHECKPOINT_BACKEND must be one of: {', '.join(valid_backends)}")

    # Check PostgreSQL config if using postgres backend
    if config.get("checkpoint_backend") == "postgres":
        if not config.get("postgres_password"):
            errors.append("POSTGRES_PASSWORD is required when using postgres backend")

    return len(errors) == 0, errors
