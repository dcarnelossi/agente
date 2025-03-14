import os

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()  # Carrega variáveis de ambiente


# Defina as variáveis de ambiente no próprio sistema ou no .env
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Certifique-se de que todas as variáveis estão definidas
if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_VERSION]):
    raise ValueError("Faltam variáveis de ambiente para configurar o AzureChatOpenAI.")

# Instância do modelo
model = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    openai_api_key=AZURE_OPENAI_API_KEY,
)
