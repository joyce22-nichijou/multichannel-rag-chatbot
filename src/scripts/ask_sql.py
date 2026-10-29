import sys

from raglab1.chatbot.sql_rag import answer_from_database
from raglab1.llm.gemini import GeminiLLM
from raglab1.retriever.sql_retriever import SqlRetriever


def main() -> None:
    question = " ".join(sys.argv[1:]) or "Which Aeldari units have toughness 4?"
    llm = GeminiLLM()
    answer, result = answer_from_database(question, SqlRetriever(llm), llm)

    print(f"Question: {question}\n")
    print(f"Generated SQL:\n{result.sql}\n")
    print(
        f"Rows returned: {len(result.rows)}{' (truncated)' if result.truncated else ''}\n"
    )
    print(answer)


if __name__ == "__main__":
    main()
