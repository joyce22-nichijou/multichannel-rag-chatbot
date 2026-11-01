import operator
from typing import Annotated, Literal

from langgraph.graph import MessagesState


class ChatState(MessagesState):
    """Data passed between the nodes of the chat graph.

    `messages` comes from MessagesState: it is the whole conversation, and LangGraph
    appends the new messages to it. The other fields describe the current turn.

    `contexts` and `sources` are filled by the retrieval nodes. With the route "both",
    the two retrieval nodes run in parallel, and operator.add concatenates their lists.
    The plan step empties both lists at the start of each turn.
    """

    question: str  # the latest message rewritten as a standalone English question
    language: str  # the language the answer must be written in
    route: Literal["rules", "database", "both", "other"]
    # Evidence given to the LLM: one string per chunk or SQL result
    contexts: Annotated[list[str], operator.add]
    # Short description of each piece of evidence, shown to the user
    sources: Annotated[list[str], operator.add]
    answer: str
