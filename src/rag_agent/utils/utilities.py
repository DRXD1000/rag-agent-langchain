"""Utility module."""

from collections.abc import Sequence

import langcodes
from fast_langdetect import LangDetectConfig, LangDetector
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)

config = LangDetectConfig(
    cache_dir="/custom/cache/path",  # Custom model cache directory
    allow_fallback=True,  # Enable fallback to small model if large model fails
)
detector = LangDetector(config)


def format_docs_for_citations(docs: Sequence[Document]) -> str:
    """Format the documents for citations.

    Args:
    ----
        docs (Sequence[Document]): Langchain documents from a vectordatabase.

    Returns:
    -------
        str: Combined documents in a format suitable for citations.

    """
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def detect_lang(text: str) -> str:
    """Detect the language of the input text.

    Args:
        text (str): The input text to analyze

    Returns:
        str: The name of the Language.

    """
    result = detector.detect(text, low_memory=False)
    return str(langcodes.get(result).language_name())


def get_chat_history(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    """Append the chat history to the messages.

    Args:
    ----
        messages (Sequence[BaseMessage]): Messages from the frontend.

    Returns:
    -------
        Sequence[BaseMessage]: Chat history as Langchain messages.

    """
    return [
        {"content": message.content, "role": message.type}
        for message in messages
        if (isinstance(message, AIMessage) and not message.tool_calls) or isinstance(message, HumanMessage)
    ]
