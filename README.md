# Visual RAG Shopping Assistant

Upload a photo of a product and get recommendations from the store catalog. Built with LangChain (LCEL), Groq, FAISS, and Streamlit.

## How it works

1. **Vision**: a Groq vision model turns the image into a short product-search description.
2. **Filter extraction**: the shopper's question is parsed into structured `SearchFilters` (price range, category) via `with_structured_output`.
3. **Hybrid retrieval**: semantic FAISS search over the whole catalog, filtered by metadata. If nothing matches the constraints, it falls back to the closest products and says so.
4. **Answer**: a grounded prompt recommends only from the retrieved products, streamed to the UI.

The FAISS index is cached in `.index/` and rebuilt automatically when `data/products.json` or the embedding model changes.

## Project layout

```
app.py                     Streamlit UI
shop_assistant/
  config.py                Settings loaded from .env
  models.py                LLM and embedding factories
  catalog.py               Product schema, catalog loading, cached FAISS index
  retrieval.py             SearchFilters + hybrid ProductRetriever
  chains.py                LCEL chains: vision, filter extraction, answer
  assistant.py             ShoppingAssistant orchestrating the pipeline
data/products.json         Product catalog (60 products, 15 categories)
tests/                     Offline pytest suite (fake LLMs and embeddings)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your GROQ_API_KEY
streamlit run app.py
```

Embeddings run locally with `fastembed` by default (free, no token). Set `EMBEDDING_PROVIDER=huggingface` and `HF_TOKEN` to use the Hugging Face Inference API instead.

## Tests

```bash
pytest
```
