import uuid
from datetime import timedelta

import pytest
from langchain_openai import ChatOpenAI
from langsmith import expect, unit

from evals.email_assistant.graph import CURRENT_TIME
from evals.email_assistant.graph import graph as assistant_graph
from evals.utils import create_openai_logprobs_classification_chain

simple_questions = [
    (
        "What's the most recent email in my inbox?",
        "Email from Jane about the agenda for our meeting tomorrow, including project report and upcoming milestones.",
    ),
    ("Has mike responded yet?", "no"),
    (
        "what time's the project kickoff?",
        (CURRENT_TIME + timedelta(days=3, hours=9)).isoformat(),
    ),
]


def classify(inputs: dict):
    return {
        "input": (
            "Evaluate the correctness of the response:\n\n"
            f"<prediction>\n{inputs['prediction']}\n</prediction>\n\n"
            "Compare against the ground truth answer:\n"
            f"<expected>\n{inputs['reference']}\n</expected>"
        )
    }


def get_score(predictions: list):
    # Output is like: [{'classification': 'incorrect', 'confidence': 1.0}]
    classes = {p["classification"]: p["confidence"] for p in predictions}
    score = None
    if "correct" in classes:
        score = classes["correct"]
    elif "incorrect" in classes:
        score = 1.0 - classes["incorrect"]

    return {"key": "correct-likelihood", "score": score}


classifier = (
    classify
    | create_openai_logprobs_classification_chain(
        ChatOpenAI(model="gpt-4-turbo"),
        {
            "correct": "The response contains the correct, expected answer.",
            "incorrect": "The response contradicts or does not recall the information in the expected answer.",
        },
    )
    | get_score
).with_config(run_name="CorrectnessClassifier")


@unit(output_keys=["expected"])
@pytest.mark.parametrize("question, expected", simple_questions)
def test_simple_questions(question: str, expected: str) -> str:
    result = assistant_graph.invoke(
        {"messages": ("user", question)},
        {"configurable": {"user_id": "vwp@langchain.com"}},
    )
    response = str(result["messages"][-1].content)
    expect.embedding_distance(response, expected)
    run_id = uuid.uuid4()
    result = classifier.invoke(
        {"prediction": response, "reference": expected}, {"run_id": run_id}
    )
    expect.score(
        result["score"], key=result["key"], source_run_id=run_id
    ).to_be_greater_than(0.5)
    return response
