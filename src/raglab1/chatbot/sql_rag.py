from raglab1.llm.gemini import GeminiLLM
from raglab1.retriever.sql_retriever import SqlResult, SqlRetriever

SYSTEM_PROMPT = """You answer questions about Warhammer 40,000 units.
Use ONLY the database query result given with the question, not your own memory.
If the result is empty, say that no matching data was found in the unit database.
Say that the information comes from the unit database (datasheets)."""


def format_result(result: SqlResult) -> str:
    """Write the query and its rows as a small text table for the prompt."""
    lines = [f"SQL query:\n{result.sql}", "", "Result:"]
    lines.append(" | ".join(result.columns))
    for row in result.rows:
        lines.append(" | ".join(str(value) for value in row))
    if result.truncated:
        lines.append(f"(only the first {len(result.rows)} rows are shown)")
    return "\n".join(lines)


def answer_from_database(
    question: str, retriever: SqlRetriever, llm: GeminiLLM
) -> tuple[str, SqlResult]:
    """Query the unit database, then let the LLM answer from the result only."""
    result = retriever.retrieve(question)
    if result.error:
        return f"I could not query the unit database ({result.error}).", result
    prompt = f"{format_result(result)}\n\nQuestion: {question}"
    return llm.generate(prompt, system=SYSTEM_PROMPT), result
