import sys

from raglab1.chatbot.text_rag import answer_question
from raglab1.llm.gemini import GeminiLLM
from raglab1.retriever.text_retriever import TextRetriever


def main() -> None:
    question = " ".join(sys.argv[1:]) or "How many points is a Strike Force game?"
    answer, chunks = answer_question(question, TextRetriever(), GeminiLLM())

    print(f"Question: {question}\n")
    print(answer)
    print("\nRetrieved chunks:")
    for chunk in chunks:
        print(f"  - page {chunk['pages']} | {' > '.join(chunk['headings'])}")


if __name__ == "__main__":
    main()
