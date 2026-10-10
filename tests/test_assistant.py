from langchain_core.language_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from shop_assistant.assistant import DEFAULT_QUESTION, ShoppingAssistant
from shop_assistant.chains import _vision_messages, format_context
from shop_assistant.retrieval import SearchFilters


def make_assistant(retriever, filters=None):
    return ShoppingAssistant(
        vision_llm=FakeListChatModel(responses=["Black over-ear wireless headphones"]),
        text_llm=FakeListChatModel(responses=["The JBL Tune 510BT is the best match."]),
        retriever=retriever,
        filter_chain=RunnableLambda(lambda _: filters or SearchFilters()),
    )


def test_end_to_end_with_budget(retriever):
    assistant = make_assistant(retriever, SearchFilters(max_price=60, category="Headphones"))
    result = assistant.retrieve(b"fake-image", "image/png", "headphones under $60")

    assert result.image_description == "Black over-ear wireless headphones"
    assert result.filters.max_price == 60
    assert all(d.metadata["price"] <= 60 for d in result.products)
    assert all(d.metadata["category"] == "Headphones" for d in result.products)
    assert "max $60" in result.filter_note

    assert assistant.answer(result) == "The JBL Tune 510BT is the best match."
    assert "".join(assistant.stream_answer(result))


def test_empty_question_skips_filter_extraction(retriever):
    assistant = make_assistant(retriever)
    result = assistant.retrieve(b"fake-image", "image/jpeg", "   ")
    assert result.filters.is_empty
    assert result.question == DEFAULT_QUESTION
    assert result.filter_note == ""


def test_filter_extraction_failure_is_non_fatal(retriever):
    assistant = make_assistant(retriever)

    def boom(_):
        raise RuntimeError("LLM down")

    assistant.filter_chain = RunnableLambda(boom)
    result = assistant.retrieve(b"fake-image", "image/jpeg", "under $100")
    assert result.filters.is_empty
    assert len(result.products) == 3


def test_vision_message_carries_mime_type(retriever):
    captured = {}
    assistant = make_assistant(retriever)
    assistant.vision_chain = RunnableLambda(lambda inputs: captured.update(inputs) or "desc")
    assistant.describe_image(b"abc", "image/webp")
    assert captured == {"image_b64": "YWJj", "mime_type": "image/webp"}

    [message] = _vision_messages(captured)
    assert message.content[1]["image_url"]["url"] == "data:image/webp;base64,YWJj"


def test_format_context(retriever):
    docs = retriever.search("keyboard").documents
    context = format_context(docs)
    assert context.startswith("[1] ")
    assert format_context([]) == "(no matching products)"
