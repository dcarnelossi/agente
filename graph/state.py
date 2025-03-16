# https://langchain-ai.github.io/langgraph/concepts/low_level/#state
from __future__ import annotations

import operator
from typing import Any, Dict, List, Sequence, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated


class AgentState(TypedDict, total=False):  # total=False torna todas as chaves opcionais
    """Define o estado global do agente no LangGraph, agora totalmente opcional."""

    user_query: str
    is_valid_query: bool
    table_schemas: Dict[str, str]
    table_samples: Dict[str, List[Dict[str, Any]]]


# 📌 Definição do estado de entrada
class InputState(TypedDict):
    messages: Annotated[Sequence[HumanMessage], add_messages]
    user_query: str
    is_relevant: bool
    table_schemas: Dict[str, str]
    sql_query: str
    query_response: List[Dict[str, Any]]
    uuid: str
    visualization: Annotated[str, operator.add]


# 📌 Definição do estado de saída
class OutputState(TypedDict):
    messages: Annotated[Sequence[AIMessage], add_messages]
    final_answer: str
    visualization: Annotated[str, operator.add]
    visualization_reason: Annotated[str, operator.add]
