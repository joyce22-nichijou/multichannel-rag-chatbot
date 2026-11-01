import argparse

from raglab1.agent.workflow import build_agent
from raglab1.evaluation.scoring import collect_answers, score
from raglab1.evaluation.testset import load_jsonl
from raglab1.llm.gemini import GeminiLLM
from raglab1.retriever.sql_retriever import SqlRetriever
from raglab1.retriever.text_retriever import TextRetriever
from raglab1.settings import settings

TEXT_COLUMNS = ["user_input", "retrieved_contexts", "response", "reference", "route"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the chat agent with RAGAS.")
    parser.add_argument(
        "--limit", type=int, help="only the first N questions (quick test)"
    )
    args = parser.parse_args()

    samples = load_jsonl(settings.testset_path)[: args.limit]
    llm = GeminiLLM()
    agent = build_agent(llm, TextRetriever(), SqlRetriever(llm))

    print(f"1/2 Asking the agent {len(samples)} questions...")
    records = collect_answers(agent, samples, settings.eval_answers_path)
    print("2/2 Scoring with RAGAS...")
    table = score(records, llm, settings.eval_results_path)

    print("\nRoutes chosen by the agent:")
    print(table["route"].value_counts().to_string())
    print("\nMean scores (0 to 1):")
    metrics = table.drop(columns=TEXT_COLUMNS)
    print(metrics.mean(numeric_only=True).round(3).to_string())
    print(f"\nDetails saved to {settings.eval_results_path}")


if __name__ == "__main__":
    main()
