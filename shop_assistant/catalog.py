import hashlib
import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field


class Product(BaseModel):
    id: int
    name: str
    brand: str
    category: str
    description: str
    price: float = Field(gt=0)
    rating: float | None = Field(default=None, ge=0, le=5)
    tags: list[str] = Field(default_factory=list)

    def to_document(self) -> Document:
        content = (
            f"Product: {self.name}\n"
            f"Brand: {self.brand}\n"
            f"Category: {self.category}\n"
            f"Description: {self.description}\n"
            f"Tags: {', '.join(self.tags)}\n"
            f"Price: ${self.price:g}"
        )
        return Document(page_content=content, metadata=self.model_dump())


def load_products(path: Path) -> list[Product]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    products = [Product.model_validate(item) for item in raw]

    ids = [p.id for p in products]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate product ids in {path}")
    return products


def _fingerprint(path: Path, embedding_id: str) -> str:
    digest = hashlib.sha256(path.read_bytes())
    digest.update(embedding_id.encode())
    return digest.hexdigest()[:16]


def build_vector_store(
    products_path: Path,
    embeddings: Embeddings,
    index_dir: Path | None = None,
    embedding_id: str = "",
) -> FAISS:
    """Build a FAISS index over the catalog, reusing a cached copy on disk when the
    catalog file and embedding model are unchanged."""
    cache_path = None
    if index_dir is not None:
        cache_path = index_dir / _fingerprint(products_path, embedding_id)
        if (cache_path / "index.faiss").exists():
            return FAISS.load_local(
                str(cache_path), embeddings, allow_dangerous_deserialization=True
            )

    docs = [p.to_document() for p in load_products(products_path)]
    store = FAISS.from_documents(docs, embeddings)

    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)
        store.save_local(str(cache_path))
    return store
