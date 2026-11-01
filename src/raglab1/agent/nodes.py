from typing import Literal

from langgraph.types import Overwrite
from pydantic import BaseModel, Field

from raglab1.agent.state import ChatState
from raglab1.chatbot.sql_rag import format_result
from raglab1.chatbot.text_rag import format_context
from raglab1.llm.gemini import GeminiLLM
from raglab1.retriever.sql_retriever import SqlRetriever
from raglab1.retriever.text_retriever import TextRetriever
from raglab1.settings import settings

PLAN_PROMPT = """You are the first step of a chatbot about the Warhammer 40,000 tabletop game.
Read the conversation and the user's latest message, then fill in four fields.

question: the latest message rewritten as one complete, standalone question in English.
Use the conversation to replace references such as "it" or "this unit" by what they mean.

language: the language the answer must be written in. This is the language of the latest
message, unless the user explicitly asks for another one: for "Traduis ta réponse en
anglais" (French for "translate your answer into English"), the language is English.

needs_rulebook: true if the answer needs a rule of the game from the Core Rules book: how
the game is played (phases, moving, Advancing, shooting, charging, fighting, terrain,
keywords, missions...).

needs_database: true if the answer needs data about a specific, named unit or faction
from the unit database: its characteristics (Move, toughness, save, wounds...), weapons,
abilities, keywords, stratagems, enhancements or points. A rules question that only
mentions a type of unit (Vehicle, Monster, Infantry...) does not need the database.

Decide the two fields separately: a question can need both, e.g. "What is the Move of the
Intercessor Squad, and how far can it move when it Advances?" (its Move from the database,
the Advance rule from the book). Both are false for greetings, thanks, questions
unrelated to Warhammer 40,000, and requests about the conversation itself, such as
translating or summarising a previous answer."""

ANSWER_PROMPT = """You are a chatbot that answers questions about the Warhammer 40,000 tabletop game.
Always answer in {language}.

When evidence is given:
- Use ONLY the evidence, not your own memory of the game: it may be outdated or wrong.
- If the evidence does not cover a part of the question, say that you could not find that
  part. Do not complete it from memory and do not guess.
- If the evidence contains both rulebook passages and database results, combine them:
  take the unit's data from the database and apply the rules of the rulebook to it.
- Cite your sources: the page for the rulebook, e.g. (p. 55), and the unit's name for the
  database, e.g. (unit database: Intercessor Squad). Only cite pages and units that
  appear in the evidence.

When no evidence is given, the message is small talk or a request about the conversation,
for example a translation: answer it using the conversation. If it is a question unrelated
to Warhammer 40,000, say politely that you only answer questions about the game."""


class Plan(BaseModel):
    """The result of the plan step. Gemini fills it in as JSON."""

    question: str = Field(
        description="The latest message as a standalone English question."
    )
    language: str = Field(description="The language the answer must be written in.")
    needs_rulebook: bool = Field(
        description="True if the answer needs a rule from the Core Rules book."
    )
    needs_database: bool = Field(
        description="True if the answer needs data about a specific unit or faction."
    )

    @property
    def route(self) -> Literal["rules", "database", "both", "other"]:
        """The route deduced from the two yes/no answers."""
        if self.needs_rulebook and self.needs_database:
            return "both"
        if self.needs_rulebook:
            return "rules"
        if self.needs_database:
            return "database"
        return "other"


def format_history(messages: list, turns: int) -> str:
    """Write the last turns of the conversation as 'User: ...' and 'Assistant: ...' lines."""
    lines = [
        f"{'User' if message.type == 'human' else 'Assistant'}: {message.content}"
        for message in messages[-2 * turns :]
    ]
    return "\n".join(lines) or "(no previous messages)"


class ChatNodes:
    """The steps of the chat graph.

    Each method receives the current state and returns only the fields it changes.
    """

    def __init__(
        self,
        llm: GeminiLLM,
        text_retriever: TextRetriever,
        sql_retriever: SqlRetriever,
    ) -> None:
        self._llm = llm
        self._text_retriever = text_retriever
        self._sql_retriever = sql_retriever

    def plan(self, state: ChatState) -> dict:
        """Rewrite the latest message in English, choose the answer language and a route."""
        *history, latest = state["messages"]
        prompt = (
            "Conversation so far:\n"
            f"{format_history(history, settings.chat_history_turns)}\n\n"
            f"Latest message: {latest.content}"
        )
        answer = self._llm.generate(prompt, system=PLAN_PROMPT, schema=Plan)
        plan = Plan.model_validate_json(answer)
        return {
            "question": plan.question,
            "language": plan.language,
            "route": plan.route,
            # Empty the evidence lists, so that nothing is left over from the previous
            # turn. Overwrite replaces the value instead of adding to it (operator.add)
            "contexts": Overwrite([]),
            "sources": Overwrite([]),
        }

    def retrieve_rules(self, state: ChatState) -> dict:
        """Search the Core Rules book (PDF channel)."""
        chunks = self._text_retriever.retrieve(state["question"])
        return {
            "contexts": [format_context([chunk]) for chunk in chunks],
            "sources": [
                f"Core Rules p. {', '.join(str(p) for p in chunk['pages'])}: "
                f"{' > '.join(chunk['headings'])}"
                for chunk in chunks
            ],
        }

    def retrieve_database(self, state: ChatState) -> dict:
        """Query the unit database (SQL channel)."""
        result = self._sql_retriever.retrieve(state["question"])
        if result.error:
            return {
                "contexts": [f"The database query failed: {result.error}"],
                "sources": [f"Unit database, query failed: {result.sql}"],
            }
        return {
            "contexts": [format_result(result)],
            "sources": [f"Unit database ({len(result.rows)} rows): {result.sql}"],
        }

    def answer(self, state: ChatState) -> dict:
        """Write the answer in the user's language, from the evidence only."""
        *history, latest = state["messages"]
        evidence = "\n\n".join(state["contexts"]) or "(no evidence)"
        prompt = (
            "Conversation so far:\n"
            f"{format_history(history, settings.chat_history_turns)}\n\n"
            f"Evidence:\n{evidence}\n\n"
            f"User's latest message: {latest.content}\n"
            f"The same message as a standalone English question: {state['question']}"
        )
        system = ANSWER_PROMPT.format(language=state["language"])
        answer = self._llm.generate(prompt, system=system)
        return {"answer": answer, "messages": [("assistant", answer)]}


def choose_route(state: ChatState) -> str | list[str]:
    """Tell the conditional edge which node(s) come after the plan step.

    For "both", the two retrieval nodes run in parallel, then the answer node runs once.
    """
    if state["route"] == "both":
        return ["rules", "database"]
    return state["route"]
