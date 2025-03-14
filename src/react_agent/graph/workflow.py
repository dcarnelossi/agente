from langgraph.graph import END, START, StateGraph
from state import InputState, OutputState

from agent import EcommerceAgent


class WorkflowManager:
    def __init__(self):
        self.agent = EcommerceAgent()

    def create_workflow(self) -> StateGraph:
        """Cria e configura o fluxo de trabalho do agente."""
        workflow = StateGraph(input=InputState, output=OutputState)

        # Definição dos nós
        workflow.add_node("interact_with_user", self.agent.interact_with_user)
        workflow.add_node("collect_user_interaction", self.agent.collect_user_interaction)
        workflow.add_node("analyze_tables", self.agent.analyze_tables)
        workflow.add_node("generate_sql", self.agent.generate_sql)
        workflow.add_node("validate_sql", self.agent.validate_sql)
        workflow.add_node("execute_sql", self.agent.execute_sql)
        workflow.add_node("generate_answer", self.agent.generate_answer)
        workflow.add_node("choose_visualization", self.agent.choose_visualization)

        def route_after_interaction(state: dict) -> str:
            """
            Se a pergunta for válida (`is_relevant=True`), segue para `analyze_tables`.
            Caso contrário, interrompe a execução e aguarda nova interação do usuário.
            """
            if state.get("is_relevant", False):  # 🔹 Agora verifica corretamente `True`
                return "analyze_tables"

            return "collect_user_interaction"

        def route_after_validation(state: dict) -> str:
            """Se a query falhar na validação, volta para a geração."""
            return "generate_sql" if state.get("retry_generate_sql") else "execute_sql"

        def route_after_execution(state: dict) -> str:
            """Se a execução falhar, volta para gerar a query."""
            return "generate_sql" if state.get("query_error") else "generate_answer"

        # Fluxo de execução
        workflow.add_edge(START, "interact_with_user")
        workflow.add_conditional_edges("interact_with_user", route_after_interaction)
        workflow.add_edge("collect_user_interaction", "interact_with_user")
        workflow.add_edge("analyze_tables", "generate_sql")
        workflow.add_edge("generate_sql", "validate_sql")
        workflow.add_conditional_edges("validate_sql", route_after_validation)
        workflow.add_conditional_edges("execute_sql", route_after_execution)
        workflow.add_edge("generate_answer", "choose_visualization")
        workflow.add_edge("choose_visualization", END)

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


graph = WorkflowManager().returnGraph()
