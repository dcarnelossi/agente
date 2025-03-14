# https://langchain-ai.github.io/langgraph/concepts/low_level/#state
from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict, total=False):  # total=False torna todas as chaves opcionais
    """Define o estado global do agente no LangGraph, agora totalmente opcional."""

    user_query: str
    is_valid_query: bool
    table_schemas: Dict[str, str]
    table_samples: Dict[str, List[Dict[str, Any]]]
