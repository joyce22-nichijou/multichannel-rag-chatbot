from typing import Literal

from langgraph.graph import MessagesState


class ChatState(MessagesState):
    """Data passed between the nodes of the chat graph.

    `messages` comes from MessagesState: it is the whole conversation, and LangGraph
    appends the new messages to it. The other fields describe the current turn and are
    overwritten at each new message.
    """

    question: str  # the latest message rewritten as a standalone English question
    language: str  # the language the answer must be written in
    route: Literal["rules", "database", "other"]
    contexts: list[str]  # evidence given to the LLM: one string per chunk or SQL result
    sources: list[str]  # short description of each piece of evidence, shown to the user
    answer: str
