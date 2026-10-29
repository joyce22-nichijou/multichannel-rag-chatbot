import threading
import time

from google import genai
from google.genai import types
from pydantic import BaseModel

from raglab1.settings import settings

# Shared by every GeminiLLM object, so the rate limit holds for the whole program
_lock = threading.Lock()
_last_call = 0.0


class GeminiLLM:
    """Generate text with a Gemini model through the Google AI Studio API."""

    def __init__(self, model: str = settings.llm_model) -> None:
        if not settings.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is missing: add it to the .env file at the project root."
            )
        self._client = genai.Client(api_key=settings.google_api_key)
        self._model = model

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        schema: type[BaseModel] | None = None,
    ) -> str:
        """Send one prompt and return the model's text answer.

        With a schema, Gemini answers with JSON that follows this schema.
        """
        _wait_for_rate_limit()
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json" if schema else None,
                response_schema=schema,
            ),
        )
        return response.text or ""


def _wait_for_rate_limit() -> None:
    """Keep at least 60 / llm_requests_per_minute seconds between two calls."""
    global _last_call
    min_interval = 60 / settings.llm_requests_per_minute
    with _lock:
        wait = _last_call + min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
