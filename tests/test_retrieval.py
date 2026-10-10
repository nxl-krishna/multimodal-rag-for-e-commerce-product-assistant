from shop_assistant.retrieval import SearchFilters


def test_unfiltered_search_returns_k(retriever):
    result = retriever.search("wireless headphones")
    assert len(result.documents) == 3
    assert not result.filters_relaxed


def test_max_price_is_applied_across_whole_catalog(retriever):
    result = retriever.search("headphones", SearchFilters(max_price=50))
    assert result.documents
    assert all(d.metadata["price"] <= 50 for d in result.documents)
    assert not result.filters_relaxed


def test_price_range_and_category(retriever):
    filters = SearchFilters(min_price=100, max_price=300, category="headphones")
    result = retriever.search("noise cancelling", filters)
    assert result.documents
    for d in result.documents:
        assert d.metadata["category"] == "Headphones"
        assert 100 <= d.metadata["price"] <= 300


def test_impossible_filters_fall_back(retriever):
    result = retriever.search("laptop", SearchFilters(max_price=1))
    assert result.filters_relaxed
    assert len(result.documents) == 3


def test_normalize_drops_unknown_category_and_swaps_bounds(retriever):
    filters = retriever.normalize(SearchFilters(category="Spaceships", min_price=500, max_price=100))
    assert filters.category is None
    assert (filters.min_price, filters.max_price) == (100, 500)


def test_normalize_drops_non_positive_prices(retriever):
    filters = retriever.normalize(SearchFilters(max_price=0, min_price=-5))
    assert filters.is_empty
