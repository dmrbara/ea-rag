from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Load environment variables from project root .env if present
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Paths:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = project_root / "data"
    artifacts_dir: Path = project_root / "artifacts"
    cache_dir: Path = artifacts_dir / "cache"
    predictions_dir: Path = artifacts_dir / "predictions"
    eval_dir: Path = artifacts_dir / "eval"


@dataclass(frozen=True)
class Models:
    embedding_model: str = "text-embedding-3-small"


@dataclass(frozen=True)
class Env:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")


paths = Paths()
models = Models()
env = Env()


