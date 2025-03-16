from typing import Any

from database import PostgresDB
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode

db = PostgresDB()


def handle_tool_error(state) -> dict:
    """
    Handles tool errors and returns a structured error message.
    """
    error = state.get("error", "Unknown error")  # Usa um fallback seguro
    last_message = state["messages"][-1]

    # Garante que tool_calls existe antes de acessar
    tool_calls = getattr(last_message, "tool_calls", [])

    if not tool_calls:
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: {repr(error)}. No tool_calls were found in the last message.",
                    tool_call_id=None,
                )
            ]
        }

    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}. Please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    return ToolNode(tools).with_fallbacks([RunnableLambda(handle_tool_error)], exception_key="error")


def format_samples_for_prompt(table_samples: dict) -> str:
    """
    Formata os exemplos de dados para serem inseridos no prompt de geração de SQL.
    Limita cada tabela a 3 exemplos para evitar sobrecarga de tokens.
    """
    formatted_samples = []

    for table, samples in table_samples.items():
        if not samples:
            continue

        # Pegamos no máximo 3 exemplos por tabela
        sample_rows = "\n".join([str(row) for row in samples[:3]])

        formatted_samples.append(f"🔹 **Exemplos de {table}**:\n{sample_rows}")

    return "\n\n".join(formatted_samples)