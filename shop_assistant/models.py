from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq

from shop_assistant.config import Settings


def get_vision_llm(settings: Settings) -> BaseChatModel:
    return ChatGroq(model=settings.vision_model, api_key=settings.groq_api_key, temperature=0.0)


def get_text_llm(settings: Settings, temperature: float = 0.3) -> BaseChatModel:
    return ChatGroq(model=settings.text_model, api_key=settings.groq_api_key, temperature=temperature)


def get_embeddings(settings: Settings) -> Embeddings:
    provider = settings.embedding_provider.lower()

    if provider == "fastembed":
        from langchain_community.embeddings import FastEmbedEmbeddings

        return FastEmbedEmbeddings(model_name=settings.embedding_model)

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        return HuggingFaceEndpointEmbeddings(
            model=settings.embedding_model,
            task="feature-extraction",
            huggingfacehub_api_token=settings.hf_token,
        )

    raise ValueError(f"Unknown EMBEDDING_PROVIDER '{settings.embedding_provider}'")
