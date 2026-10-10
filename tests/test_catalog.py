import json

import pytest
from pydantic import ValidationError

from shop_assistant.catalog import Product, build_vector_store, load_products


def test_catalog_is_valid(products_path, raw_products):
    products = load_products(products_path)
    assert len(products) == len(raw_products) >= 50
    assert len({p.category for p in products}) >= 10


def test_document_contains_searchable_fields():
    product = Product(
        id=1, name="Test Phone", brand="Acme", category="Smartphone",
        description="A phone", price=199.5, tags=["android"],
    )
    doc = product.to_document()
    assert "Test Phone" in doc.page_content
    assert "Acme" in doc.page_content
    assert "$199.5" in doc.page_content
    assert doc.metadata["price"] == 199.5


def test_rejects_invalid_price():
    with pytest.raises(ValidationError):
        Product(id=1, name="x", brand="x", category="x", description="x", price=0)


def test_rejects_duplicate_ids(tmp_path):
    item = {"id": 1, "name": "x", "brand": "x", "category": "x", "description": "x", "price": 1}
    path = tmp_path / "products.json"
    path.write_text(json.dumps([item, item]))
    with pytest.raises(ValueError, match="Duplicate"):
        load_products(path)


def test_index_is_cached_and_invalidated(tmp_path, products_path, embeddings):
    catalog = tmp_path / "products.json"
    catalog.write_bytes(products_path.read_bytes())
    index_dir = tmp_path / "index"

    first = build_vector_store(catalog, embeddings, index_dir=index_dir, embedding_id="fake")
    assert len(list(index_dir.iterdir())) == 1

    second = build_vector_store(catalog, embeddings, index_dir=index_dir, embedding_id="fake")
    assert second.index.ntotal == first.index.ntotal
    assert len(list(index_dir.iterdir())) == 1

    build_vector_store(catalog, embeddings, index_dir=index_dir, embedding_id="other-model")
    assert len(list(index_dir.iterdir())) == 2
