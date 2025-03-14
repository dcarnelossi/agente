from agent.graph.state import AgentState
from database import PostgresDB
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from llm import model  # Importamos o modelo configurado
from prompt import format_answer_prompt, sql_prompt


def classify_query(state: AgentState) -> dict:
    """
    Classifica a pergunta do usuário e decide se o agente pode respondê-la.
    Se for relevante para e-commerce, continua o fluxo normal.
    Caso contrário, responde ao usuário e encerra a conversa.
    """
    user_query = state["messages"][-1].content  # Última mensagem do usuário

    classification_prompt = f"""
    Você é um assistente de IA especializado em e-commerce.
    Sua tarefa é classificar a pergunta do usuário como 'SIM' (se for relacionada a vendas, produtos ou faturamento de uma loja online) ou 'NÃO' (se for irrelevante).
    
    Pergunta: "{user_query}"

    Responda apenas com 'SIM' ou 'NÃO'.
    """

    response = model.invoke([HumanMessage(content=classification_prompt)]).content.strip().upper()

    if response == "SIM":
        return {"messages": [AIMessage(content="Essa é uma pergunta válida sobre e-commerce. Vou processá-la.")]}

    return {
        "messages": [
            AIMessage(
                content="Desculpe, só posso responder a perguntas relacionadas a vendas, produtos ou faturamento de e-commerces."
            )
        ]
    }


def analyze_tables(state: AgentState) -> AgentState:
    """
    Analisa a estrutura e exemplos de dados das tabelas e separa corretamente o esquema e os exemplos.

    :param state: Estado do agente.
    :return: Estado atualizado com esquemas e exemplos de dados organizados corretamente.
    """
    db = PostgresDB()  # Conexão com o banco
    toolkit = SQLDatabaseToolkit(db=db.db, llm=model)

    tools = toolkit.get_tools()
    get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

    tables = ["orders_ia", "orders_items_ia"]

    # Inicializa os dicionários no estado, se ainda não existirem
    state.setdefault("table_schemas", {})
    state.setdefault("table_samples", {})

    for table in tables:
        try:
            full_info = get_schema_tool.invoke(table)
            # 🔹 **Armazenamos separadamente**
            state["table_schemas"][table] = full_info
            print(state["table_schemas"][table])

        except Exception:
            state["table_schemas"][table] = "Erro ao obter esquema"

    return state


def generate_sql(state: AgentState) -> AgentState:
    """
    Gera uma query SQL baseada na pergunta do usuário e no esquema do banco.
    Apenas gera a query, sem executá-la.

    :param state: Estado do agente contendo a pergunta do usuário e informações das tabelas.
    :return: Estado atualizado com a query SQL gerada.
    """
    user_question = state.get("user_query", "")
    table_schemas = state.get("table_schemas", {})

    if not user_question:
        state["sql_query"] = "Erro: Nenhuma pergunta foi fornecida."
        return state

    if not table_schemas:
        state["sql_query"] = "Erro: Nenhuma informação de tabela disponível para gerar a consulta."
        return state

    # Criamos um contexto SQL apenas com os esquemas das tabelas
    schema_context = "\n".join([f"Table {table}: {schema}" for table, schema in table_schemas.items()])
    print(schema_context)

    # Criamos o prompt e pedimos ao LLM para gerar a query SQL
    sql_prompt_text = sql_prompt(schema_context, user_question)
    sql_query = model.invoke([HumanMessage(content=sql_prompt_text)]).content.strip()

    if not sql_query.lower().startswith("select"):
        state["sql_query"] = f"Erro: A query gerada não é uma consulta SELECT válida.\nQuery: {sql_query}"
        return state

    state["sql_query"] = sql_query
    print(sql_query)
    return state


@tool
def execute_sql(query: str) -> list:
    """
    Executa uma query SQL no banco de dados PostgreSQL.
    Retorna os resultados em formato de lista.
    """
    db = PostgresDB()

    result = db.run_no_throw(query)

    if not result:
        return ["Erro: Falha ao executar a consulta. Por favor, reescreva sua query e tente novamente."]

    # 🔹 Garantimos que o resultado seja sempre uma lista
    if isinstance(result, str):
        try:
            result = eval(result)  # Converte string de lista em lista real
        except Exception:
            result = ["Erro ao processar o resultado da query."]

    return result if isinstance(result, list) else [result]  # Garante que sempre retorne uma lista


def generate_answer(state: AgentState) -> AgentState:
    """
    Gera uma resposta para o usuário com base na query SQL e nos dados retornados.

    :param state: Estado do agente contendo a pergunta original, a query SQL gerada e os resultados da query.
    :return: Estado atualizado com a resposta final.
    """
    user_question = state.get("user_query", "")
    sql_query = state.get("sql_query", "")
    query_result = state.get("query_response", "")

    if (
        not query_result
        or query_result == "Erro: Falha ao executar a consulta. Por favor, reescreva sua query e tente novamente."
    ):
        state["final_answer"] = "Desculpe, não foi possível obter uma resposta para sua pergunta."
        return state

    # 🔹 Gera o prompt formatado para o LLM
    answer_prompt = format_answer_prompt(user_question, sql_query, query_result)

    # 🔹 Chama o LLM para gerar a resposta final
    final_answer = model.invoke([HumanMessage(content=answer_prompt)]).content.strip()

    # 🔹 Atualiza o estado com a resposta gerada
    state["final_answer"] = final_answer
    print(state["final_answer"])

    return state
