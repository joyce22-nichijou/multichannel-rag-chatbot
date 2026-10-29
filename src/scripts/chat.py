import uuid

from google.genai.errors import APIError
from pydantic import ValidationError
from rich.console import Console
from rich.markdown import Markdown

from raglab1.agent.workflow import build_agent
from raglab1.llm.gemini import GeminiLLM
from raglab1.retriever.sql_retriever import SqlRetriever
from raglab1.retriever.text_retriever import TextRetriever

WELCOME = """[bold]Warhammer 40,000 assistant[/bold]
Ask about the rules or the units, in any language.
Commands: /new (start a new conversation), /quit
"""


def main() -> None:
    console = Console()
    with console.status("Loading models..."):
        llm = GeminiLLM()
        agent = build_agent(llm, TextRetriever(), SqlRetriever(llm))
    thread_id = str(uuid.uuid4())
    console.print(WELCOME)

    while True:
        try:
            message = console.input("[bold cyan]You> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not message:
            continue
        if message == "/quit":
            break
        if message == "/new":
            thread_id = str(uuid.uuid4())
            console.print("[dim]New conversation.[/dim]\n")
            continue

        try:
            with console.status("Thinking..."):
                state = agent.invoke(
                    {"messages": [("user", message)]},
                    config={"configurable": {"thread_id": thread_id}},
                )
        # Gemini errors (e.g. quota exceeded) or an invalid plan: keep the chat running
        except (APIError, ValidationError) as error:
            console.print(f"[red]Error: {error}[/red]\n")
            continue

        console.print(Markdown(state["answer"]))
        console.print(
            f"[dim]route: {state['route']} | question: {state['question']}[/dim]"
        )
        for source in state["sources"]:
            console.print(f"[dim]  - {source}[/dim]")
        console.print()


if __name__ == "__main__":
    main()
