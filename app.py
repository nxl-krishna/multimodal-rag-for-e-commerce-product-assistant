import streamlit as st

from shop_assistant import ShoppingAssistant
from shop_assistant.catalog import build_vector_store
from shop_assistant.config import Settings
from shop_assistant.models import get_embeddings, get_text_llm, get_vision_llm
from shop_assistant.retrieval import ProductRetriever

st.set_page_config(page_title="Visual RAG Shopping Assistant", page_icon="🛒", layout="wide")

settings = Settings()

with st.sidebar:
    st.header("Settings")
    if not settings.groq_api_key:
        settings.groq_api_key = st.text_input("Groq API Key", type="password")
    if settings.embedding_provider == "huggingface" and not settings.hf_token:
        settings.hf_token = st.text_input("HuggingFace Token", type="password")
    settings.top_k = st.slider("Products to compare", min_value=1, max_value=6, value=settings.top_k)
    st.caption(f"Embeddings: `{settings.embedding_provider}` · `{settings.embedding_model}`")

if settings.missing_keys:
    st.info(f"⚠️ Please provide: {', '.join(settings.missing_keys)}")
    st.stop()


@st.cache_resource(show_spinner="📦 Indexing product catalog...")
def load_vector_store(provider: str, model: str, hf_token: str | None):
    return build_vector_store(
        settings.products_path,
        get_embeddings(settings),
        index_dir=settings.index_dir,
        embedding_id=f"{provider}:{model}",
    )


try:
    vector_store = load_vector_store(settings.embedding_provider, settings.embedding_model, settings.hf_token)
except FileNotFoundError:
    st.error(f"❌ Product catalog not found at `{settings.products_path}`")
    st.stop()
except Exception as e:
    st.error(f"Error building the product index: {e}")
    st.stop()

assistant = ShoppingAssistant(
    vision_llm=get_vision_llm(settings),
    text_llm=get_text_llm(settings),
    filter_llm=get_text_llm(settings, temperature=0.0),
    retriever=ProductRetriever(vector_store, k=settings.top_k),
)

st.title("🛒 Visual RAG Shopping Assistant")
st.markdown("Upload a photo of a product, and I'll find similar items in our inventory.")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "webp"])
    if uploaded_file:
        st.image(uploaded_file, caption="Your Upload", width="stretch")

with col2:
    if uploaded_file and uploaded_file.size > settings.max_image_bytes:
        st.error(f"Image is too large. Please upload an image under {settings.max_image_bytes // (1024 * 1024)}MB.")
    elif uploaded_file:
        st.subheader("Analysis")
        user_query = st.text_input(
            "Question (optional)", placeholder="e.g., Something similar under $150 for gaming?"
        )

        if st.button("🔍 Analyze Product", type="primary"):
            try:
                with st.spinner("👀 Analyzing image and searching inventory..."):
                    result = assistant.retrieve(uploaded_file.getvalue(), uploaded_file.type, user_query)

                with st.expander("Image description"):
                    st.write(result.image_description)

                if not result.filters.is_empty:
                    st.info(f"🔎 Filters applied: {result.filters.describe()}")
                if result.filters_relaxed:
                    st.warning("No products matched your constraints. Showing the closest alternatives.")

                st.markdown("### Recommendation")
                st.write_stream(assistant.stream_answer(result))

                st.markdown("### Matched products")
                for doc in result.products:
                    m = doc.metadata
                    with st.container(border=True):
                        st.markdown(f"**{m['name']}** · {m['brand']} · {m['category']}")
                        rating = f" · ⭐ {m['rating']}" if m.get("rating") else ""
                        st.markdown(f"**${m['price']:g}**{rating}")
                        st.caption(m["description"])
            except Exception as e:
                st.error(f"Error: {e}")
