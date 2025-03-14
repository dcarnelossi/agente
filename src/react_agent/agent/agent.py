from database import PostgresDB
from langchain.schema import AIMessage, HumanMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import QuerySQLCheckerTool
from llm import model
from prompt import format_answer_prompt, get_classification_prompt, sql_prompt


class EcommerceAgent:
    def __init__(self):
        self.db = PostgresDB()
        self.toolkit = SQLDatabaseToolkit(db=self.db.db, llm=model)
        self.query_checker = QuerySQLCheckerTool(db=self.db.db, llm=model)  # 🔹 Instancia o validador SQL

    def classify_query(self, state: dict) -> dict:
        """
        Classifica a consulta e adiciona uma resposta ao usuário.
        """
        user_query = state["messages"][-1].content  # Última mensagem do usuário
        classification_prompt = get_classification_prompt(user_query)

        response = model.invoke([HumanMessage(content=classification_prompt)]).content.strip()
        is_valid = response.upper() == "SIM"

        new_message = AIMessage(
            content="✅ Essa é uma pergunta válida sobre e-commerce. Vou processá-la."
            if is_valid
            else "❌ Desculpe, só posso responder perguntas sobre vendas, produtos ou faturamento."
        )

        # 🔹 Atualiza o estado sem sobrescrever mensagens
        state["messages"].append(new_message)
        state["user_query"] = user_query

        return state | {"is_valid_query": is_valid}

    def analyze_tables(self, state: dict) -> dict:
        """
        Obtém informações das tabelas e adiciona ao estado.
        """
        state["messages"].append(AIMessage(content="🔍 Obtendo informações das tabelas relevantes..."))

        tools = self.toolkit.get_tools()
        get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

        tables = ["orders_ia", "orders_items_ia"]
        table_schemas = {}

        for table in tables:
            try:
                full_info = get_schema_tool.invoke(table)
                table_schemas[table] = full_info
            except Exception:
                table_schemas[table] = "Erro ao obter esquema"

        return state | {"table_schemas": table_schemas}

    def generate_sql(self, state: dict) -> dict:
        """
        Gera uma consulta SQL para responder à pergunta do usuário.
        Se houver erro, adiciona uma mensagem no chat pedindo a correção.
        """
        user_query = state["user_query"]
        table_schemas = state.get("table_schemas", {})
        query_error = state.get("query_error")  # Verifica se há erro anterior

        if not table_schemas:
            return state | {"sql_query": "Erro: Nenhuma informação de tabela disponível."}

        # 🔹 Formata o contexto do banco de dados
        schema_context = "\n".join([f"Table {table}: {schema}" for table, schema in table_schemas.items()])

        # 🔹 Se for a primeira tentativa, adicionamos o prompt original
        if not query_error:
            sql_prompt_text = sql_prompt(schema_context, user_query)
            state["messages"].append(HumanMessage(content=sql_prompt_text))
        else:
            # 🔹 Se houve erro, adicionamos uma nova mensagem pedindo correção
            state["messages"].append(
                HumanMessage(content=f"Sua query retornou com esse erro: {query_error}. Por favor, corrija.")
            )

        # 🔹 Invocamos o LLM passando o histórico de mensagens
        llm_response = model.invoke(state["messages"])

        sql_query = llm_response.content.strip()

        if not sql_query.lower().startswith("select"):
            return state | {"sql_query": f"Erro: A query gerada não é válida.\nQuery: {sql_query}"}

        # 🔹 Adicionamos a resposta do LLM no histórico
        state["messages"].append(AIMessage(content=sql_query))

        return state | {
            "sql_query": sql_query,
            "query_error": None,  # 🔹 Resetamos o erro após a correção
            "retry_generate_sql": False,  # 🔹 Resetamos a flag para evitar loops infinitos
        }

    def validate_sql(self, state: dict) -> dict:
        """
        Valida a query SQL antes da execução usando `QuerySQLCheckerTool`.
        """
        state["messages"].append(AIMessage(content="🔎 Validando a query SQL..."))

        query = state.get("sql_query", "")
        validation_result = self.query_checker.run(query)  # 🔹 Valida a SQL

        # 🔹 Se a validação encontrar erro, retorna para gerar uma nova query
        if "ERROR:" in validation_result or "SQL state" in validation_result:
            state["messages"].append(AIMessage(content=f"❌ Erro na validação da query: {validation_result}"))
            return state | {"sql_error": validation_result, "retry_generate_sql": True}

        state["messages"].append(AIMessage(content="✅ Query SQL validada com sucesso!"))
        return state | {"sql_error": None}  # 🔹 Reseta qualquer erro anterior

    def execute_sql(self, state: dict) -> dict:
        """
        Executa a consulta SQL no banco de dados e retorna os resultados.
        Se houver erro, volta para o LLM como uma nova mensagem de chat pedindo correção.
        """
        state["messages"].append(AIMessage(content="⏳ Executando a query no banco de dados..."))

        query = state.get("sql_query", "")
        result = self.db.run_no_throw(query)  # Retorna sempre uma string

        # 🔹 Se a resposta começa com "ERROR:", consideramos uma falha na execução
        if result.startswith("ERROR:") or "SQL state:" in result:
            # 🔹 Ao invés de sobrescrever o prompt, adicionamos uma nova mensagem no chat
            state["messages"].append(
                HumanMessage(content=f"Sua query retornou com esse erro: {result}. Por favor, corrija-a.")
            )

            return state | {
                "query_error": result,
                "retry_generate_sql": True,  # 🔹 Apenas ativamos se houver erro
            }

        state["messages"].append(AIMessage(content="✅ Consulta SQL executada com sucesso."))

        return state | {
            "query_response": result,
            "retry_generate_sql": False,  # 🔹 Resetamos para evitar loops
        }

    def generate_answer(self, state: dict) -> dict:
        """
        Gera a resposta final para o usuário baseada na consulta SQL e nos dados retornados.
        """
        state["messages"].append(AIMessage(content="✅ Processando os resultados da consulta..."))

        sql_query = state.get("sql_query", "")
        query_response = state.get("query_response", [])

        if not query_response or "Erro" in query_response[0]:
            return state | {"final_answer": "❌ Desculpe, não foi possível obter uma resposta."}

        answer_prompt = format_answer_prompt(state["user_query"], sql_query, query_response)
        final_answer = model.invoke([HumanMessage(content=answer_prompt)]).content.strip()

        state["messages"].append(AIMessage(content=final_answer))

        return state | {"final_answer": final_answer}
