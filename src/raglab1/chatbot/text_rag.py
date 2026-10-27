from raglab1.llm.gemini import GeminiLLM
from raglab1.retriever.text_retriever import TextRetriever

SYSTEM_PROMPT = """You answer questions about the Warhammer 40,000 Core Rules.
Use ONLY the context provided with the question. Do not use your own memory of the
rules: they may be outdated or wrong.
If the context does not contain the answer, say that you could not find it in the
rulebook. Do not guess.
After each piece of information, cite the page it comes from, e.g. (p. 55)."""


def format_context(chunks: list[dict]) -> str:
    """Write each chunk as '[headings | page X]' followed by its text."""
    blocks = []
    for chunk in chunks:
        headings = " > ".join(chunk["headings"])
        pages = ", ".join(str(p) for p in chunk["pages"])
        blocks.append(f"[{headings} | page {pages}]\n{chunk['text']}")
    return "\n\n".join(blocks)


def answer_question(
    question: str, retriever: TextRetriever, llm: GeminiLLM
) -> tuple[str, list[dict]]:
    """Retrieve the relevant chunks, then let the LLM answer from them only."""
    chunks = retriever.retrieve(question)
    prompt = f"Context:\n{format_context(chunks)}\n\nQuestion: {question}"
    return llm.generate(prompt, system=SYSTEM_PROMPT), chunks
