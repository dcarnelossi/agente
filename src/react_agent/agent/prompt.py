def get_classification_prompt(user_query: str) -> str:
    """
    Retorna o prompt para classificar se uma pergunta é relacionada a e-commerce.
    """
    return f"""
    Você é um assistente especializado em e-commerce.
    Verifique se a seguinte pergunta está relacionada a vendas, produtos ou faturamento.
    Responda apenas com 'SIM' ou 'NÃO'.

    Pergunta: "{user_query}"
    """


def sql_prompt(schema_context: str, user_question: str) -> str:
    """
    Retorna um prompt estruturado para gerar queries SQL seguras e otimizadas no PostgreSQL.
    """
    prompt = f"""
    Você é um especialista em postgresql com alta atenção aos detalhes.

    Dada uma pergunta de entrada, gere uma **consulta SQL sintaticamente correta** para o banco de dados PostgreSQL.

    ⚠️ **REGRAS IMPORTANTES** ⚠️
    - **NÃO** faça chamadas a ferramentas além da execução da query final.
    - **NÃO** realize operações de modificação de dados (`INSERT`, `UPDATE`, `DELETE`, `DROP`, etc.).
    - Sempre **limite os resultados a no máximo 5 registros**, a menos que o usuário especifique um número maior.
    - **Ordene os resultados por uma coluna relevante** para garantir que os exemplos mais significativos sejam retornados.
    - **Nunca selecione todas as colunas (`SELECT *`)**, apenas os campos necessários para responder à pergunta do usuário.
    - **Se a execução da query falhar, gere uma nova versão corrigida da query** e tente novamente.
    - **Se a consulta retornar um conjunto vazio**, tente reformular a consulta para obter um resultado significativo.
    - **NÃO invente informações** caso os dados não estejam disponíveis para responder à pergunta.

    ⚠️ **CUIDADOS AO ESCREVER A QUERY SQL** ⚠️
    1. **Use apenas os nomes de colunas e tabelas disponíveis no esquema abaixo.**  
       - **Não consulte colunas inexistentes.**
       - **Verifique em qual tabela está cada coluna.**
    2. **Se a pergunta envolve "hoje", use `CURRENT_DATE` no PostgreSQL.**
    3. **Erros comuns que devem ser evitados**:
       - Usar `NOT IN` com valores `NULL` (pode resultar em comportamento inesperado).
       - Usar `UNION` quando `UNION ALL` seria mais adequado.
       - Usar `BETWEEN` para intervalos que deveriam ser exclusivos (`> ... AND < ...`).
       - Mismatch de tipos de dados em comparações (`VARCHAR` vs. `INTEGER`).
       - Identificadores devem ser corretamente escapados (`"column_name"` quando necessário).
       - Funções devem ser usadas corretamente (com a quantidade certa de argumentos).
       - Conversões de tipo (`CAST()`) devem ser feitas corretamente.
       - Chaves estrangeiras devem ser usadas corretamente em `JOINs`.

    ⚠️ **Se houver qualquer erro ou problema na query SQL, corrija e reescreva antes de executar.**  
    ⚠️ **Se a query estiver correta, apenas gere e retorne a query final.**  

    ### **📌 Esquema do banco de dados**
    {schema_context}

    ### **📌 Pergunta do usuário**
    "{user_question}"

    ⚠️ **Apenas gere a query SQL final. NÃO inclua explicações ou comentários adicionais no output.**
    """
    return prompt


def format_answer_prompt(user_question: str, sql_query: str, query_result: str) -> str:
    """
    Gera um prompt estruturado para o LLM responder ao usuário com base nos resultados da query SQL.
    Inclui formatação aprimorada dos dados para melhor legibilidade.
    """
    return f"""
    Você é um assistente especializado em análise de dados e SQL.
    Sua tarefa é responder à pergunta do usuário **de maneira clara e objetiva**, usando os resultados de uma consulta SQL.

    🔹 **Pergunta do usuário**:
    "{user_question}"

    🔹 **Query SQL executada**:
    {sql_query}

    🔹 **Resultado da Query**:
    {query_result}

    ✅ **Baseando-se apenas nesses dados, forneça uma resposta clara e bem formatada.**
    ✅ **Formatos recomendados**:
       - **Números grandes** → Use separadores de milhar (`1.000.000,00` em vez de `1000000.00`).
       - **Datas** → Use o formato `DD/MM/AAAA` (exemplo: `17/07/2024` em vez de `2024-07-17`).
       - **Moedas** → Exiba valores monetários no formato `R$ X.XXX,XX` (exemplo: `R$ 1.200,00` em vez de `1200.00`).

    ❌ **Não faça suposições. Se os dados não forem suficientes, informe isso ao usuário.**
    ❌ **Não repita a query SQL na resposta. Apenas forneça a informação processada.**
    """
