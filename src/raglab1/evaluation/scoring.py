import json
import uuid
from pathlib import Path

import pandas as pd
from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.run_config import RunConfig

from raglab1.evaluation.ragas_adapters import GeminiRagasLLM
from raglab1.evaluation.testset import load_jsonl
from raglab1.llm.gemini import GeminiLLM


def collect_answers(agent, samples: list[dict], path: Path) -> list[dict]:
    """Ask the chat agent every test question and record what it answered.

    Each answer is saved at once: after a crash (e.g. quota exceeded), running again
    only asks the questions that are not in the file yet.
    """
    saved = {r["user_input"]: r for r in load_jsonl(path)} if path.exists() else {}
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with path.open("a", encoding="utf-8") as f:
        for sample in samples:
            question = sample["user_input"]
            if question not in saved:
                # A new thread_id for each question: no memory between test questions
                state = agent.invoke(
                    {"messages": [("user", question)]},
                    config={"configurable": {"thread_id": str(uuid.uuid4())}},
                )
                saved[question] = {
                    "user_input": question,
                    "reference": sample["reference"],
                    "response": state["answer"],
                    # The evidence the agent really used: none if retrieval was skipped
                    "retrieved_contexts": state["contexts"] or ["(no context)"],
                    "route": state["route"],
                }
                f.write(json.dumps(saved[question], ensure_ascii=False) + "\n")
                f.flush()
            records.append(saved[question])
    return records


def score(records: list[dict], llm: GeminiLLM, path: Path) -> pd.DataFrame:
    """Score the answers with four RAGAS metrics, Gemini acting as the judge."""
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": r["user_input"],
                "response": r["response"],
                "retrieved_contexts": r["retrieved_contexts"],
                "reference": r["reference"],
            }
            for r in records
        ]
    )
    result = evaluate(
        dataset,
        metrics=[
            LLMContextPrecisionWithReference(),  # retrieval: useful chunks ranked first?
            LLMContextRecall(),  # retrieval: was the needed information found?
            Faithfulness(),  # generation: is every claim supported by the chunks?
            FactualCorrectness(),  # generation: does it agree with the reference?
        ],
        llm=GeminiRagasLLM(llm),
        run_config=RunConfig(max_workers=2, timeout=300),
    )
    table = result.to_pandas()
    table["route"] = [r["route"] for r in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8")
    return table
