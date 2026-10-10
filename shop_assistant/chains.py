from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from shop_assistant.retrieval import SearchFilters

# The description is embedded as the search query, so it must be short plain prose:
# markdown and "not visible" filler measurably degrade retrieval with small embedding models.
VISION_PROMPT = (
    "Describe the product in this image as a short product-search query of one or two "
    "plain-text sentences. Start with the product type, then mention color, form factor, "
    "materials, and any visible brand or model. Only state what you can actually see; "
    "skip anything that is not visible. No markdown, no headings, no lists."
)

FILTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract shopping search filters from a customer's question. "
            "Only set a field when the customer states it explicitly. "
            "Known categories: {categories}.",
        ),
        ("human", "{question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful e-commerce shopping assistant. Recommend products ONLY from "
            "the store inventory provided. Never invent products, prices, or specs. "
            "Lead with the single best match and why it fits, then briefly mention the "
            "alternatives. Always quote prices. If none of the inventory resembles the "
            "uploaded item, say so honestly.",
        ),
        (
            "human",
            "Uploaded image description:\n{image_description}\n\n"
            "Customer question: {question}\n\n"
            "{filter_note}"
            "Store inventory matches:\n{context}",
        ),
    ]
)


def _vision_messages(inputs: dict) -> list[HumanMessage]:
    return [
        HumanMessage(
            content=[
                {"type": "text", "text": VISION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{inputs['mime_type']};base64,{inputs['image_b64']}"},
                },
            ]
        )
    ]


def build_vision_chain(llm: BaseChatModel) -> Runnable:
    """Input: {"image_b64": str, "mime_type": str} -> product description string."""
    return RunnableLambda(_vision_messages) | llm | StrOutputParser()


def build_filter_chain(llm: BaseChatModel) -> Runnable:
    """Input: {"question": str, "categories": str} -> SearchFilters."""
    return FILTER_PROMPT | llm.with_structured_output(SearchFilters)


def build_answer_chain(llm: BaseChatModel) -> Runnable:
    """Input: image_description, question, filter_note, context -> answer string."""
    return ANSWER_PROMPT | llm | StrOutputParser()


def format_context(documents: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(documents, start=1):
        m = doc.metadata
        blocks.append(f"[{i}] {m['name']} — ${m['price']:g}\n{doc.page_content}")
    return "\n\n".join(blocks) if blocks else "(no matching products)"
