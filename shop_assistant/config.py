import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value else default


@dataclass
class Settings:
    groq_api_key: str | None = field(default_factory=lambda: _env("GROQ_API_KEY"))
    hf_token: str | None = field(default_factory=lambda: _env("HF_TOKEN"))

    # "fastembed" runs locally for free; "huggingface" uses the HF Inference API (needs credits)
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "fastembed"))
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    vision_model: str = field(default_factory=lambda: _env("VISION_MODEL", "qwen/qwen3.8-27b"))
    text_model: str = field(default_factory=lambda: _env("TEXT_MODEL", "openai/gpt-oss-120b"))

    products_path: Path = field(
        default_factory=lambda: Path(_env("PRODUCTS_PATH", str(PROJECT_ROOT / "data" / "products.json")))
    )
    index_dir: Path = field(
        default_factory=lambda: Path(_env("INDEX_DIR", str(PROJECT_ROOT / ".index")))
    )

    top_k: int = 3
    max_image_bytes: int = 3 * 1024 * 1024  # Groq rejects base64 image payloads over 4MB

    @property
    def missing_keys(self) -> list[str]:
        missing = []
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if self.embedding_provider == "huggingface" and not self.hf_token:
            missing.append("HF_TOKEN")
        return missing
