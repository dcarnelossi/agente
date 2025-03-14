import unittest

from agent.graph.state import AgentState
from database import PostgresDB
from tools import analyze_tables, classify_query, execute_sql, generate_answer, generate_sql


def default_state() -> AgentState:
    """Retorna um estado inicial válido para o agente."""
    return {
        "user_query": None,
        "is_valid_query": None,
        "table_schemas": {},
        "table_samples": {},
    }


class TestDatabaseConnection(unittest.TestCase):
    """Testes para verificar a conexão com o banco de dados."""

    @classmethod
    def setUpClass(cls):
        """Configuração inicial: cria uma conexão ao banco de dados."""
        cls.db = PostgresDB()

    def test_get_tables(self):
        """Testa se é possível listar as tabelas disponíveis."""
        tables = self.db.get_tables()
        print(f"Tabelas disponíveis: {tables}")
        self.assertIsInstance(tables, list)

    def test_get_context(self):
        """Testa a obtenção do contexto do banco de dados para LangChain."""
        context = self.db.get_context()
        print("Contexto do banco de dados:", context)

        # Verifica se o contexto é um dicionário
        self.assertIsInstance(context, dict)
        self.assertIn("table_info", context)  # Verifica se contém informações das tabelas
        self.assertIsInstance(context["table_info"], str)  # Garante que a informação da tabela seja uma string

    def test_get_table_info(self):
        """Testa a obtenção de informações sobre uma tabela específica."""
        table_info = self.db.get_table_info(["orders_ia"])
        print("Informações da tabela:", table_info)

        # O retorno é uma string contendo o esquema da tabela
        self.assertIsInstance(table_info, str)  # Agora verificamos se o retorno é string
        self.assertTrue("CREATE TABLE" in table_info)  # Garante que é um esquema válido


class TestClassifyQuery(unittest.TestCase):
    """Testes para verificar se a função classify_query está funcionando corretamente."""

    def test_valid_queries(self):
        """Testa consultas válidas relacionadas a vendas e faturamento."""
        self.assertTrue(classify_query("Qual foi o faturamento da loja no último mês?"))
        self.assertTrue(classify_query("Quantos produtos foram vendidos ontem?"))
        self.assertTrue(classify_query("Quais são os produtos mais vendidos da loja?"))

    def test_invalid_queries(self):
        """Testa consultas não relacionadas ao escopo do e-commerce."""
        self.assertFalse(classify_query("Qual o clima em São Paulo hoje?"))
        self.assertFalse(classify_query("Quem ganhou a Copa do Mundo de 2018?"))
        self.assertFalse(classify_query("Me conte uma piada sobre programadores."))


class TestAnalyzeTables(unittest.TestCase):
    """Testes para verificar se a análise das tabelas funciona corretamente."""

    def test_analyze_tables(self):
        """Testa se as tabelas 'orders_ia' e 'orders_items_ia' são analisadas corretamente."""

        # Criamos um estado inicial vazio
        state: AgentState = {}

        # Chamamos a função que atualiza o estado
        updated_state = analyze_tables(state)

        expected_tables = ["orders_ia", "orders_items_ia"]

        for table in expected_tables:
            self.assertIn(table, updated_state["table_schemas"])
            self.assertIn(table, updated_state["table_samples"])
            self.assertIsInstance(updated_state["table_schemas"][table], str)
            self.assertIsInstance(updated_state["table_samples"][table], list)


class TestGenerateSQL(unittest.TestCase):
    """Testes para verificar se a geração e execução de SQL funcionam corretamente usando tabelas reais."""

    @classmethod
    def setUpClass(cls):
        """Configuração inicial: Analisa as tabelas antes dos testes."""
        cls.state: AgentState = {}
        cls.state = analyze_tables(cls.state)  # Preenche o estado com tabelas reais

    def test_generate_valid_sql(self):
        """Testa se a função gera uma query SQL corretamente com base nas tabelas reais."""
        self.state["user_query"] = "Qual foi o faturamento total?"

        updated_state = generate_sql(self.state)

        self.assertIn("sql_query", updated_state)
        self.assertTrue(updated_state["sql_query"].strip().lower().startswith("select"))

    # def test_generate_sql_no_question(self):
    #     """Testa se a função retorna erro quando não há pergunta."""
    #     self.state["user_query"] = ""

    #     updated_state = generate_sql(self.state)

    #     self.assertEqual(updated_state["sql_query"], "Erro: Nenhuma pergunta foi fornecida.")

    # def test_generate_sql_no_schema(self):
    #     """Testa se a função retorna erro quando não há informações de tabelas."""
    #     state: AgentState = {
    #         "user_query": "Quantos pedidos foram feitos?",
    #         "table_schemas": {},  # Nenhuma informação de tabela
    #     }

    #     updated_state = generate_sql(state)

    #     self.assertEqual(
    #         updated_state["sql_query"], "Erro: Nenhuma informação de tabela disponível para gerar a consulta."
    #     )

    # def test_execute_valid_sql(self):
    #     """Testa a execução de uma query gerada corretamente."""
    #     self.state["user_query"] = "Qual foi o faturamento total?"
    #     updated_state = generate_sql(self.state)

    #     sql_query = updated_state.get("sql_query", "")
    #     if sql_query.lower().startswith("select"):
    #         result = execute_sql(sql_query)
    #         self.assertIsInstance(result, list)  # Esperamos que seja uma lista de resultados
    #     else:
    #         self.fail(f"A query gerada não é válida: {sql_query}")

    # def test_execute_invalid_sql(self):
    #     """Testa se a execução falha corretamente ao rodar uma query inválida."""
    #     invalid_query = "SELECT unknown_column FROM unknown_table"

    #     result = execute_sql(invalid_query)
    #     self.assertEqual(
    #         result, "Erro: Falha ao executar a consulta. Por favor, reescreva sua query e tente novamente."
    #     )


class TestExecuteSQL(unittest.TestCase):
    """Testa a execução da query SQL gerada."""

    @classmethod
    def setUpClass(cls):
        """Configuração inicial: Obtém tabelas e gera uma query válida."""
        cls.state: AgentState = {}

        # 🔹 1. Obtém as tabelas reais antes dos testes
        cls.state = analyze_tables(cls.state)

        # 🔹 2. Define a pergunta e gera a query SQL
        cls.state["user_query"] = "Qual foi o faturamento total?"
        cls.state = generate_sql(cls.state)

    def test_execute_valid_sql(self):
        """Testa a execução da query SQL gerada."""

        sql_query = self.state.get("sql_query", "")
        self.assertTrue(sql_query.lower().startswith("select"), f"Query inválida: {sql_query}")

        # 🔹 3. Executa a query gerada
        result = execute_sql.invoke(sql_query)

        # 🔹 4. Valida se a consulta retornou resultados
        self.assertIsInstance(result, list, "O resultado da query não é uma lista.")
        self.assertTrue(len(result) > 0, "A query não retornou nenhum dado.")


class TestFullAgentFlow(unittest.TestCase):
    """Testa o fluxo completo do agente, desde a análise do banco até a resposta final."""

    @classmethod
    def setUpClass(cls):
        """Configuração inicial: Obtém tabelas, gera query, executa consulta e cria resposta."""
        cls.state: AgentState = {}

        # 🔹 1. Obtém os esquemas das tabelas e exemplos de dados
        cls.state = analyze_tables(cls.state)

        # 🔹 2. Define a pergunta do usuário
        cls.state["user_query"] = "Quais são os 3 métodos de pagamento mais utilizados pelos clientes?"

        # 🔹 3. Gera a query SQL baseada na pergunta
        cls.state = generate_sql(cls.state)
        sql_query = cls.state.get("sql_query", "")

        # 🔹 4. Executa a query gerada, se for válida
        if sql_query.lower().startswith("select"):
            cls.state["query_response"] = execute_sql.invoke(sql_query)
        else:
            cls.state["query_response"] = "Erro ao gerar query."

        # 🔹 5. Gera a resposta final baseada nos dados retornados
        cls.state = generate_answer(cls.state)

    def test_sql_generation(self):
        """Testa se a query SQL foi gerada corretamente."""
        self.assertIn("sql_query", self.state)
        self.assertTrue(
            self.state["sql_query"].strip().lower().startswith("select"), f"Query inválida: {self.state['sql_query']}"
        )

    def test_sql_execution(self):
        """Testa se a query SQL foi executada corretamente e retornou resultados."""
        self.assertIn("query_response", self.state)
        self.assertIsInstance(self.state["query_response"], list, "O resultado da query não é uma lista.")
        self.assertTrue(len(self.state["query_response"]) > 0, "A query não retornou nenhum dado.")

    def test_final_answer_generation(self):
        """Testa se a resposta final foi gerada corretamente com base nos dados."""
        self.assertIn("final_answer", self.state)
        self.assertIsInstance(self.state["final_answer"], str)
        self.assertTrue(len(self.state["final_answer"]) > 0, "A resposta final não pode estar vazia.")


if __name__ == "__main__":
    unittest.main()

