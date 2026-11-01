import asyncio
import os

from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM
from ragas.run_config import RunConfig

from raglab1.llm.gemini import GeminiLLM
from raglab1.storage.embedder import Embedder

# Do not send usage analytics to RAGAS (read lazily, so setting it here is enough)
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")


class GeminiRagasLLM(BaseRagasLLM):
    """Let RAGAS call Gemini through our GeminiLLM, so the rate limit still applies."""

    def __init__(self, llm: GeminiLLM) -> None:
        super().__init__()
        self._llm = llm

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: list[str] | None = None,
        callbacks=None,
    ) -> LLMResult:
        text = prompt.to_string()
        generations = [Generation(text=self._llm.generate(text)) for _ in range(n)]
        return LLMResult(generations=[generations])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks=None,
    ) -> LLMResult:
        # RAGAS runs its jobs concurrently: each call runs in a thread,
        # and the lock in gemini.py keeps them under the rate limit
        return await asyncio.to_thread(self.generate_text, prompt, n)

    def is_finished(self, response: LLMResult) -> bool:
        return True


class EmbedderRagasEmbeddings(BaseRagasEmbeddings):
    """Let RAGAS compute embeddings with our sentence-transformers model."""

    def __init__(self, embedder: Embedder) -> None:
        super().__init__()
        # The base class does not set run_config, but RAGAS reads it when embedding
        self.set_run_config(RunConfig())
        self._embedder = embedder

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)
