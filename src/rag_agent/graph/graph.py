"""Langchain Langgraph."""

import numpy as np
from langchain_core.messages import (
    convert_to_messages,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    PromptTemplate,
)

from rag_agent.llms.models import init_reranker, initialize_ollama_llm
from rag_agent.retriever.qdrant_hybrid_retriever import init_qdrant_hybrid_retriever
from rag_agent.utils.models import AgentState
from rag_agent.utils.prompts import REPHRASE_TEMPLATE
from rag_agent.utils.utilities import get_chat_history

retriever = init_qdrant_hybrid_retriever()
reranker = init_reranker()
model = initialize_ollama_llm("gemma3:12b")


def retrieve_documents(state: AgentState) -> AgentState:
    """Retrieve documents from the retriever.

    Args:
    ----
        state (AgentState): Graph State.

    Returns:
    -------
        AgentState: Modified Graph State.

    """
    messages = convert_to_messages(state["messages"])
    query = messages[-1].content
    relevant_documents = retriever.invoke(query)
    return {"query": query, "documents": relevant_documents}


def retrieve_documents_with_chat_history(state: AgentState) -> AgentState:
    """Retrieve documents from the retriever with chat history.

    Args:
    ----
        state (AgentState): Graph State.

    Returns:
    -------
        AgentState: Modified Graph State.

    """
    condense_queston_prompt = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (condense_queston_prompt | model | StrOutputParser()).with_config(
        run_name="CondenseQuestion",
    )

    messages = convert_to_messages(state["messages"])
    query = messages[-1].content
    retriever_with_condensed_question = condense_question_chain | retriever
    relevant_documents = retriever_with_condensed_question.invoke({"question": query, "chat_history": get_chat_history(messages[:-1])})
    return {"query": query, "documents": relevant_documents}


def reranking(state: AgentState, limit: int = 4) -> AgentState:
    """Rerank the documents to return best documents."""
    scores = reranker.compute_score(state["documents"])

    best_indices = np.argsort(scores)[-limit:][::-1]

    return {"documents": [state["documents"][idx] for idx in best_indices]}
