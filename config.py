"""Central configuration for DocuMind.

Every tunable value lives here. Values can be overridden with the DOCUMIND_*
environment variables documented in README.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

APP_NAME = "DocuMind"
APP_TAGLINE = "Intelligent Document Understanding System"
APP_VERSION = "0.1.0"

PROJECT_ROOT = Path(__file__).resolve().parent

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
CLOUD_SYNC_ENV_VARS = ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")
CLOUD_SYNC_FOLDER_MARKERS = ("onedrive", "dropbox", "google drive", "googledrive", "icloud")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def is_cloud_synced(path: Path) -> bool:
    """Return True if ``path`` is inside a folder a sync client uploads."""
    resolved = path.resolve()
    for var in CLOUD_SYNC_ENV_VARS:
        root = os.environ.get(var)
        if root and resolved.is_relative_to(Path(root).resolve()):
            return True
    return any(
        part.lower().startswith(marker)
        for part in resolved.parts
        for marker in CLOUD_SYNC_FOLDER_MARKERS
    )


def resolve_data_dir(project_root: Path = PROJECT_ROOT) -> Path:
    """Choose where documents and indexes are stored.

    Order: DOCUMIND_DATA_DIR, then ``data/`` in the project. If the project sits
    in a cloud-synced folder, the per-user local app data folder replaces
    ``data/``, because syncing would upload user documents and can corrupt the
    live SQLite database. An explicit DOCUMIND_DATA_DIR is always respected.
    """
    override = os.environ.get("DOCUMIND_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if is_cloud_synced(project_root):
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = Path(local_app_data) if local_app_data else Path.home() / ".local" / "share"
        return base / APP_NAME / "data"
    return project_root / "data"


def validate_local_url(url: str) -> str:
    """Return ``url`` if it points at this machine, otherwise raise ValueError."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in LOOPBACK_HOSTS:
        raise ValueError(
            f"Ollama URL must be a loopback address such as http://localhost:11434, got {url!r}"
        )
    return url.rstrip("/")


@dataclass(frozen=True)
class Paths:
    data_dir: Path = field(default_factory=resolve_data_dir)

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def indexes_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def database_path(self) -> Path:
        return self.data_dir / "documind.db"


@dataclass(frozen=True)
class IngestionConfig:
    allowed_extensions: tuple[str, ...] = (".pdf", ".docx", ".txt")
    # Keep in step with server.maxUploadSize in .streamlit/config.toml.
    max_file_size_mb: int = field(
        default_factory=lambda: _env_int("DOCUMIND_MAX_FILE_SIZE_MB", 50)
    )
    # Limit on a DOCX's decompressed size, which guards against zip bombs.
    max_docx_uncompressed_mb: int = 300
    # A PDF page with less text than this, and images covering at least
    # min_image_coverage of its area, is treated as a scanned image.
    min_chars_per_page: int = 20
    min_image_coverage: float = 0.5
    # A PDF with at least this share of scanned pages is reported as needing OCR.
    scanned_page_ratio: float = 0.5


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 100
    # Close a chunk early at a paragraph end once it is this full, rather than
    # splitting the next paragraph across two chunks.
    paragraph_break_ratio: float = 0.6
    # Each chunk is searched as passages of about this size, because one vector for a
    # 500-token chunk blurs together every subject in it.
    passage_size_tokens: int = 128

    def __post_init__(self) -> None:
        if not 0 <= self.chunk_overlap_tokens < self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be in [0, chunk_size_tokens)")


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = field(
        default_factory=lambda: os.environ.get(
            "DOCUMIND_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    dimension: int = 384
    # all-MiniLM-L6-v2 truncates input beyond this. Any longer text is embedded in
    # windows of this size and then pooled, so none of it is ignored.
    max_input_tokens: int = 256
    batch_size: int = 32
    # CPU keeps the laptop GPU's limited VRAM free for Ollama.
    device: str = "cpu"


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    # Cosine floor below which evidence counts as insufficient. Calibrated on 28 test
    # questions: relevant passages scored 0.28 or higher, unrelated ones 0.19 or lower.
    min_similarity: float = 0.25


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = field(
        default_factory=lambda: validate_local_url(
            os.environ.get("DOCUMIND_OLLAMA_URL", "http://localhost:11434")
        )
    )
    default_model: str = field(
        default_factory=lambda: os.environ.get("DOCUMIND_OLLAMA_MODEL", "qwen2.5:3b")
    )
    temperature: float = 0.2
    max_output_tokens: int = 768
    # Extra output allowed for a thinking model's reasoning, which is thrown away.
    thinking_tokens: int = 2048
    context_window_tokens: int = 8192
    # Input limits per prompt, leaving room in the window for instructions and the reply.
    summary_input_tokens: int = 2200
    qa_context_tokens: int = 2400
    connect_timeout_seconds: float = 3.0
    generate_timeout_seconds: float = 300.0


@dataclass(frozen=True)
class AnalysisConfig:
    max_keywords: int = 20
    max_topics: int = 5
    terms_per_topic: int = 8
    sections_per_topic: int = 3
    similar_passage_threshold: float = 0.75
    # A passage scoring below this against every passage of the other document is unique.
    unique_passage_threshold: float = 0.5
    # Sentences this close, ignoring figures, are treated as the same statement.
    statement_match_threshold: float = 0.7
    # Overall similarity is this share semantic, and the rest lexical.
    semantic_weight: float = 0.7
    comparison_list_limit: int = 10
    heatmap_cells: int = 30
    reading_words_per_minute: int = 200


@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


settings = Settings()


def ensure_directories(paths: Paths = settings.paths) -> None:
    """Create the local storage folders if they do not exist."""
    for directory in (
        paths.uploads_dir,
        paths.processed_dir,
        paths.indexes_dir,
        paths.logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
