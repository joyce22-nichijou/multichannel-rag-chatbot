import json
import random
from pathlib import Path

from langchain_core.documents import Document
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator

from raglab1.evaluation.ragas_adapters import EmbedderRagasEmbeddings, GeminiRagasLLM
from raglab1.llm.gemini import GeminiLLM
from raglab1.storage.embedder import Embedder

# Shorter chunks rarely contain enough information for a question
MIN_CHUNK_CHARS = 300


def select_chunks(chunks: list[dict], n: int, seed: int) -> list[dict]:
    """Pick n random chunks that are long enough. The seed makes the choice repeatable."""
    candidates = [chunk for chunk in chunks if len(chunk["text"]) >= MIN_CHUNK_CHARS]
    return random.Random(seed).sample(candidates, min(n, len(candidates)))


def generate_testset(
    chunks: list[dict], size: int, llm: GeminiLLM, embedder: Embedder
) -> list[dict]:
    """Let RAGAS write `size` questions with reference answers from the given chunks."""
    documents = [
        Document(
            page_content=chunk["text_for_embedding"],
            metadata={"chunk_id": chunk["chunk_id"], "pages": chunk["pages"]},
        )
        for chunk in chunks
    ]
    generator = TestsetGenerator(
        llm=GeminiRagasLLM(llm), embedding_model=EmbedderRagasEmbeddings(embedder)
    )
    testset = generator.generate_with_chunks(
        documents,
        testset_size=size,
        run_config=RunConfig(max_workers=2, timeout=300),
        # A failed LLM call skips one question instead of stopping everything
        raise_exceptions=False,
    )
    return [
        {
            "user_input": row["user_input"],
            "reference": row["reference"],
            "reference_contexts": list(row["reference_contexts"]),
            "synthesizer": row["synthesizer_name"],
        }
        for row in testset.to_list()
    ]


def save_jsonl(records: list[dict], path: Path) -> None:
    """Write one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    """Read a file written by save_jsonl."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
