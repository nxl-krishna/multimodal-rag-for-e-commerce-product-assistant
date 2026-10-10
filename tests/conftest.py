import json

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from shop_assistant.catalog import build_vector_store
from shop_assistant.config import PROJECT_ROOT
from shop_assistant.retrieval import ProductRetriever

PRODUCTS_PATH = PROJECT_ROOT / "data" / "products.json"


@pytest.fixture(scope="session")
def products_path():
    return PRODUCTS_PATH


@pytest.fixture(scope="session")
def raw_products():
    return json.loads(PRODUCTS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def embeddings():
    return DeterministicFakeEmbedding(size=64)


@pytest.fixture(scope="session")
def vector_store(products_path, embeddings):
    return build_vector_store(products_path, embeddings)


@pytest.fixture
def retriever(vector_store):
    return ProductRetriever(vector_store, k=3)
