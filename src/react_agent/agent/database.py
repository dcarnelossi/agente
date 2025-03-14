import os

from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine


class PostgresDB:
    """Wrapper para gerenciar a conexão com PostgreSQL usando Langchain, suportando schemas dinâmicos."""

    def __init__(self, schema: str = None):
        """Inicializa a conexão com o PostgreSQL, carregando configurações do .env."""
        load_dotenv()  # Carrega variáveis de ambiente

        self.host = os.getenv("PG_HOST")
        self.port = os.getenv("PG_PORT", "5432")
        self.database = os.getenv("PG_DATABASE")
        self.user = os.getenv("PG_USER")
        self.password = os.getenv("PG_PASSWORD")
        self.schema = schema or os.getenv("PG_SCHEMA", "public")  # Usa schema padrão caso não seja informado

        # Cria a conexão ao banco de dados
        self.db = self._connect()

    def _connect(self):
        """Cria o engine do SQLAlchemy e instancia o SQLDatabase com suporte a schemas e engine_args."""
        connection_string = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

        # Configurações adicionais do SQLAlchemy para otimizar conexões
        engine_args = {
            "pool_size": 10,  # Define o tamanho do pool de conexões
            "max_overflow": 5,  # Número máximo de conexões extras
            "echo": False,  # Define se logs SQL devem ser exibidos
        }

        # Cria o engine do SQLAlchemy
        engine = create_engine(connection_string, **engine_args)

        # Instancia o SQLDatabase com o schema informado
        return SQLDatabase(engine=engine, schema=self.schema)

    def change_schema(self, new_schema: str):
        """Altera dinamicamente o schema do banco de dados sem precisar recriar a conexão."""
        self.schema = new_schema
        self.db = SQLDatabase(engine=self.db.engine, schema=self.schema)
        print(f"🔄 Schema alterado para '{self.schema}'")

    def get_tables(self):
        """Retorna a lista de tabelas disponíveis no schema atual."""
        try:
            return self.db.get_usable_table_names()
        except Exception as e:
            print("❌ Erro ao buscar tabelas:", str(e))
            return []

    def run_query(self, query: str):
        """Executa uma consulta SQL e retorna o resultado."""
        try:
            return self.db.run(query)
        except Exception as e:
            print("❌ Erro ao executar query:", str(e))
            return None

    def __getattr__(self, name):
        """Intercepta chamadas de métodos desconhecidos e redireciona para SQLDatabase."""
        return getattr(self.db, name)
