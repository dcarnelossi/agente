from typing import Any, Dict, List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from agent import EcommerceAgent


# 📌 Definição do estado de entrada
class InputState(TypedDict):
    messages: List[HumanMessage]
    user_query: str
    table_schemas: Dict[str, str]
    sql_query: str
    query_response: List[Dict[str, Any]]
    uuid: str


# 📌 Definição do estado de saída
class OutputState(TypedDict):
    messages: List[AIMessage]
    final_answer: str


class WorkflowManager:
    def __init__(self):
        self.agent = EcommerceAgent()

    def create_workflow(self) -> StateGraph:
        """Cria e configura o fluxo de trabalho."""
        workflow = StateGraph(input=InputState, output=OutputState)

        # Adicionando nós ao fluxo de trabalho
        workflow.add_node("classify_query", self.agent.classify_query)
        workflow.add_node("analyze_tables", self.agent.analyze_tables)
        workflow.add_node("generate_sql", self.agent.generate_sql)
        workflow.add_node("validate_sql", self.agent.validate_sql)  # 🔹 Validação antes da execução
        workflow.add_node("execute_sql", self.agent.execute_sql)
        workflow.add_node("generate_answer", self.agent.generate_answer)

        def route_after_validation(state: dict) -> str:
            """Se a query falhar na validação ou na execução, retorna para gerar a SQL."""
            if state.get("retry_generate_sql"):
                return "generate_sql"
            return "execute_sql"

        def route_after_execution(state: dict) -> str:
            """Se a execução da SQL falhar, volta para generate_sql para corrigir a query."""
            if state.get("query_error"):
                return "generate_sql"
            return "generate_answer"

        # 📌 Definição das conexões entre os nós
        workflow.add_edge(START, "classify_query")
        workflow.add_edge("classify_query", "analyze_tables")
        workflow.add_edge("analyze_tables", "generate_sql")
        workflow.add_edge("generate_sql", "validate_sql")
        workflow.add_conditional_edges("validate_sql", route_after_validation)
        workflow.add_conditional_edges(
            "execute_sql", route_after_execution
        )  # 🔹 Agora executa o roteamento após execução
        workflow.add_edge("generate_answer", END)

        return workflow

    def run_sql_agent(self, input_state: dict) -> dict:
        """Run the SQL agent workflow and return the formatted answer and visualization recommendation."""
        app = self.create_workflow().compile()
        result = app.invoke(input_state)
        return {
            "final_answer": result["final_answer"],
            "messages": result["messages"],
        }

    def returnGraph(self):
        return self.create_workflow().compile()
