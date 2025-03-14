import uuid

from langchain_core.messages import HumanMessage
from workflow import WorkflowManager

if __name__ == "__main__":
    user_question = "Quais são os 5 produtos mais vendidos em quantidade nos últimos 3 meses?"
    user_uuid = str(uuid.uuid4())  # Gera um UUID único para a requisição

    input_state = {"messages": [HumanMessage(content=user_question)], "uuid": user_uuid}

    # 📌 Criar instância do WorkflowManager
    workflow_manager = WorkflowManager()

    print("🚀 Iniciando workflow...")

    # 📌 Executando o agente SQL
    output = workflow_manager.run_sql_agent(input_state)

    print("\n✅ Workflow Finalizado!")
    print("Resposta Final:", output["final_answer"])

    print("\n📌 Mensagens do fluxo:")
    for message in output["messages"]:
        print(f"{message.__class__.__name__}: {message.content}")
