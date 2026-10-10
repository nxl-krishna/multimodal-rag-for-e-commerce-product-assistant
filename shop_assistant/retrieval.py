from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    """Structured constraints extracted from the shopper's question."""

    max_price: float | None = Field(
        default=None, description="Upper price limit in USD if the user states a budget, e.g. 'under 100'."
    )
    min_price: float | None = Field(
        default=None, description="Lower price limit in USD if the user asks for something above a price."
    )
    category: str | None = Field(
        default=None,
        description="Product category only if the user explicitly names one. Must be one of the known categories.",
    )

    @property
    def is_empty(self) -> bool:
        return self.max_price is None and self.min_price is None and self.category is None

    def matches(self, metadata: dict) -> bool:
        price = metadata.get("price")
        if self.max_price is not None and (price is None or price > self.max_price):
            return False
        if self.min_price is not None and (price is None or price < self.min_price):
            return False
        if self.category is not None and str(metadata.get("category", "")).lower() != self.category.lower():
            return False
        return True

    def describe(self) -> str:
        parts = []
        if self.category:
            parts.append(f"category: {self.category}")
        if self.min_price is not None:
            parts.append(f"min ${self.min_price:g}")
        if self.max_price is not None:
            parts.append(f"max ${self.max_price:g}")
        return ", ".join(parts)


@dataclass
class RetrievalResult:
    documents: list[Document]
    filters_relaxed: bool = False


class ProductRetriever:
    """Hybrid retriever: semantic search over the full catalog, then metadata filtering."""

    def __init__(self, vector_store: FAISS, k: int = 3):
        self.vector_store = vector_store
        self.k = k
        docs = vector_store.docstore._dict.values()
        self.categories = sorted({d.metadata["category"] for d in docs})

    def normalize(self, filters: SearchFilters) -> SearchFilters:
        """Drop a category the LLM invented and fix swapped price bounds."""
        filters = filters.model_copy()
        if filters.category:
            known = {c.lower(): c for c in self.categories}
            filters.category = known.get(filters.category.lower())
        if filters.max_price is not None and filters.max_price <= 0:
            filters.max_price = None
        if filters.min_price is not None and filters.min_price <= 0:
            filters.min_price = None
        if (
            filters.min_price is not None
            and filters.max_price is not None
            and filters.min_price > filters.max_price
        ):
            filters.min_price, filters.max_price = filters.max_price, filters.min_price
        return filters

    def search(self, query: str, filters: SearchFilters | None = None) -> RetrievalResult:
        filters = self.normalize(filters) if filters else SearchFilters()

        if filters.is_empty:
            return RetrievalResult(self.vector_store.similarity_search(query, k=self.k))

        filtered = self.vector_store.similarity_search(
            query,
            k=self.k,
            filter=filters.matches,
            fetch_k=self.vector_store.index.ntotal,
        )
        if filtered:
            return RetrievalResult(filtered)

        return RetrievalResult(
            self.vector_store.similarity_search(query, k=self.k), filters_relaxed=True
        )
