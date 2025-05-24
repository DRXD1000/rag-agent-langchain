from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """State Class for Langgraph-Stategraph."""

    query: str
    documents: list[Document]
    messages: Annotated[list[BaseMessage], add_messages]
    refined_query: str
