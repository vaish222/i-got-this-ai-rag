from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .grounded_generation import GENERATION_MODE_STRICT_FILTER, GENERATION_MODES


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    pinecone_index_name: str
    pinecone_namespace: str
    pinecone_cloud: str
    pinecone_region: str
    embedding_model: str
    chat_model: str
    ollama_base_url: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    reference_date: str
    timezone: str
    generation_mode: str = GENERATION_MODE_STRICT_FILTER

    @classmethod
    def from_environment(cls, project_root: Path) -> "Settings":
        project_root = project_root.resolve()
        raw_data_dir = Path(os.getenv("RAG_DATA_DIR", "data/sample")).expanduser()
        data_dir = raw_data_dir if raw_data_dir.is_absolute() else project_root / raw_data_dir
        settings = cls(
            project_root=project_root,
            data_dir=data_dir,
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "i-got-this-phase-1"),
            pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "baseline"),
            pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
            pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
            embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma"),
            chat_model=os.getenv("OLLAMA_CHAT_MODEL", "gemma3:1b"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "75")),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
            reference_date=os.getenv("RAG_REFERENCE_DATE", "2026-08-20"),
            timezone=os.getenv("RAG_TIMEZONE", "America/Los_Angeles"),
            generation_mode=os.getenv(
                "RAG_GENERATION_MODE",
                GENERATION_MODE_STRICT_FILTER,
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Knowledge-base directory does not exist: {self.data_dir}")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,44}", self.pinecone_index_name):
            raise ValueError(
                "PINECONE_INDEX_NAME must contain lowercase letters, numbers, or hyphens "
                "and be at most 45 characters."
            )
        if not self.pinecone_namespace.strip():
            raise ValueError("PINECONE_NAMESPACE cannot be empty.")
        if self.chunk_size <= 0:
            raise ValueError("RAG_CHUNK_SIZE must be positive.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be non-negative and smaller than RAG_CHUNK_SIZE.")
        if self.top_k < 5:
            raise ValueError("RAG_TOP_K must be at least 5 so evaluation can calculate Recall@5.")
        if self.generation_mode not in GENERATION_MODES:
            supported = ", ".join(sorted(GENERATION_MODES))
            raise ValueError(f"RAG_GENERATION_MODE must be one of: {supported}.")

    def public_config(self) -> dict[str, Any]:
        config = asdict(self)
        config["project_root"] = self.project_root.as_posix()
        config["data_dir"] = self.data_dir.as_posix()
        return config
