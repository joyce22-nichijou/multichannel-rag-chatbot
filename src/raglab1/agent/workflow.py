from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from raglab1.agent.nodes import ChatNodes, choose_route
from raglab1.agent.state import ChatState
from raglab1.llm.gemini import GeminiLLM
from raglab1.retriever.sql_retriever import SqlRetriever
from raglab1.retriever.text_retriever import TextRetriever


def build_agent(
    llm: GeminiLLM, text_retriever: TextRetriever, sql_retriever: SqlRetriever
):
    """Build the chat graph.

    START -> plan -> retrieve_rules ----> answer -> END
                  -> retrieve_database -> answer
                  -> answer                (route "other": no retrieval)
    """
    nodes = ChatNodes(llm, text_retriever, sql_retriever)
    graph = StateGraph(ChatState)

    graph.add_node("plan", nodes.plan)
    graph.add_node("retrieve_rules", nodes.retrieve_rules)
    graph.add_node("retrieve_database", nodes.retrieve_database)
    graph.add_node("answer", nodes.answer)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan",
        choose_route,
        {
            "rules": "retrieve_rules",
            "database": "retrieve_database",
            "other": "answer",
        },
    )
    graph.add_edge("retrieve_rules", "answer")
    graph.add_edge("retrieve_database", "answer")
    graph.add_edge("answer", END)

    # The checkpointer keeps the state of each conversation (thread_id) in memory
    return graph.compile(checkpointer=InMemorySaver())
