from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}
FRONT_MATTER_PATTERN = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)


def json_safe(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items() if item is not None}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    match = FRONT_MATTER_PATTERN.match(text)
    if not match:
        return {}, text
    metadata = yaml.safe_load(match.group(1)) or {}
    if not isinstance(metadata, dict):
        raise ValueError("Document front matter must be a YAML mapping.")
    return json_safe(metadata), text[match.end() :]


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def base_metadata(path: Path, project_root: Path) -> dict[str, Any]:
    relative_path = path.relative_to(project_root).as_posix()
    domain = path.parent.name
    fallback_id = re.sub(r"[^a-z0-9]+", "_", f"{domain}_{path.stem}".lower()).strip("_")
    return {
        "document_id": fallback_id,
        "document_title": path.stem.replace("_", " ").title(),
        "domain": domain,
        "document_type": path.suffix.lstrip(".").lower(),
        "source_path": relative_path,
        "file_name": path.name,
        "file_type": path.suffix.lstrip(".").lower(),
    }


def load_file(path: Path, project_root: Path) -> list[Document]:
    common_metadata = base_metadata(path, project_root)
    if path.suffix.lower() in {".md", ".txt"}:
        loaded = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()
        front_matter, body = parse_front_matter(loaded[0].page_content)
        content = clean_text(body)
        metadata = json_safe({**common_metadata, **front_matter})
        return [Document(page_content=content, metadata=metadata)] if content else []

    pages = PyPDFLoader(str(path)).load()
    documents: list[Document] = []
    for page in pages:
        content = clean_text(page.page_content)
        if not content:
            continue
        page_number = int(page.metadata.get("page", 0)) + 1
        metadata = {**common_metadata, "page_number": page_number}
        documents.append(Document(page_content=content, metadata=json_safe(metadata)))
    return documents


def corpus_paths(data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def load_corpus(data_dir: Path, project_root: Path) -> list[Document]:
    paths = corpus_paths(data_dir)
    if not paths:
        raise ValueError(f"No supported documents found under {data_dir}")
    documents = [document for path in paths for document in load_file(path, project_root)]
    if not documents:
        raise ValueError("Supported files were found, but none contained extractable text.")
    return documents


def corpus_fingerprint(data_dir: Path, project_root: Path) -> dict[str, Any]:
    manifest = [
        {
            "path": path.relative_to(project_root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in corpus_paths(data_dir)
    ]
    canonical = "\n".join(f"{item['path']}:{item['sha256']}" for item in manifest).encode()
    return {
        "document_count": len(manifest),
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "files": manifest,
    }


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    chunk_counts: Counter[str] = Counter()
    for chunk in chunks:
        document_id = str(chunk.metadata["document_id"])
        chunk_index = chunk_counts[document_id]
        chunk_counts[document_id] += 1
        chunk.metadata["chunk_index"] = chunk_index
        chunk.metadata["chunk_id"] = f"{document_id}::chunk_{chunk_index:03d}"
        chunk.metadata["chunk_size"] = chunk_size
        chunk.metadata["chunk_overlap"] = chunk_overlap
    if not chunks:
        raise ValueError("Chunking produced no content.")
    return chunks


def chunk_fingerprint(chunks: list[Document]) -> dict[str, Any]:
    manifest = [
        {
            "chunk_id": str(chunk.metadata["chunk_id"]),
            "document_id": str(chunk.metadata["document_id"]),
            "content_sha256": hashlib.sha256(chunk.page_content.encode()).hexdigest(),
        }
        for chunk in chunks
    ]
    canonical = "\n".join(
        f"{item['chunk_id']}:{item['document_id']}:{item['content_sha256']}" for item in manifest
    ).encode()
    return {
        "chunk_count": len(manifest),
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "chunks": manifest,
    }

