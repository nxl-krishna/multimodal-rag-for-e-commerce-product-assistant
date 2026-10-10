import base64
import logging
from collections.abc import Iterator
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from shop_assistant.chains import (
    build_answer_chain,
    build_filter_chain,
    build_vision_chain,
    format_context,
)
from shop_assistant.retrieval import ProductRetriever, SearchFilters

logger = logging.getLogger(__name__)

DEFAULT_QUESTION = "What is this, and which of your products is the closest match?"


@dataclass
class AssistantResult:
    image_description: str
    question: str
    filters: SearchFilters
    products: list[Document]
    filters_relaxed: bool

    @property
    def filter_note(self) -> str:
        if self.filters.is_empty:
            return ""
        if self.filters_relaxed:
            return (
                f"Note: no products matched the customer's constraints ({self.filters.describe()}); "
                "these are the closest alternatives, so tell the customer that.\n\n"
            )
        return f"All products below satisfy the customer's constraints ({self.filters.describe()}).\n\n"


class ShoppingAssistant:
    def __init__(
        self,
        vision_llm: BaseChatModel,
        text_llm: BaseChatModel,
        retriever: ProductRetriever,
        filter_llm: BaseChatModel | None = None,
        filter_chain: Runnable | None = None,
    ):
        self.retriever = retriever
        self.vision_chain = build_vision_chain(vision_llm)
        self.filter_chain = filter_chain or build_filter_chain(filter_llm or text_llm)
        self.answer_chain = build_answer_chain(text_llm)

    def describe_image(self, image_bytes: bytes, mime_type: str) -> str:
        return self.vision_chain.invoke(
            {"image_b64": base64.b64encode(image_bytes).decode("utf-8"), "mime_type": mime_type}
        )

    def extract_filters(self, question: str) -> SearchFilters:
        if not question.strip():
            return SearchFilters()
        try:
            return self.filter_chain.invoke(
                {"question": question, "categories": ", ".join(self.retriever.categories)}
            )
        except Exception:
            logger.warning("Filter extraction failed; searching without filters", exc_info=True)
            return SearchFilters()

    def retrieve(self, image_bytes: bytes, mime_type: str, question: str = "") -> AssistantResult:
        description = self.describe_image(image_bytes, mime_type)
        filters = self.retriever.normalize(self.extract_filters(question))
        result = self.retriever.search(f"{description}\n{question}".strip(), filters)
        return AssistantResult(
            image_description=description,
            question=question.strip() or DEFAULT_QUESTION,
            filters=filters,
            products=result.documents,
            filters_relaxed=result.filters_relaxed,
        )

    def _answer_inputs(self, result: AssistantResult) -> dict:
        return {
            "image_description": result.image_description,
            "question": result.question,
            "filter_note": result.filter_note,
            "context": format_context(result.products),
        }

    def answer(self, result: AssistantResult) -> str:
        return self.answer_chain.invoke(self._answer_inputs(result))

    def stream_answer(self, result: AssistantResult) -> Iterator[str]:
        return self.answer_chain.stream(self._answer_inputs(result))
