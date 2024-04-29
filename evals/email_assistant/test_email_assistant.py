import functools
import inspect
import uuid
from datetime import datetime, timedelta
from typing import Callable

import langsmith
import pytest
from langchain_openai import ChatOpenAI
from langsmith import expect, traceable, unit
from langsmith.run_helpers import get_current_run_tree
from evals.email_assistant.db import (
    CURRENT_TIME,
    get_weekday,
)
from evals.email_assistant.graph import (
    create_graph,
    search_calendar_events,
    search_emails,
)
from evals.utils import create_openai_logprobs_classification_chain

assistant_graph = create_graph()
simple_questions = [
    (
        "What's the most recent email in my inbox?",
        "Email from Jane about the agenda for our meeting tomorrow, including project report and upcoming milestones.",
    ),
    ("Has mike responded yet to my most recent email?", "no"),
    (
        "I think john@langchain.com messaged me sometime earlier today or yesterday. what was its subject?",
        "Budget Approval. Mark as incorrect if budget approval is not mentioned, correct otherwise.",
    ),
    (
        "What is the end time of the event titled 'Team Meeting'?",
        "Sometime around: "
        + (CURRENT_TIME + timedelta(days=1, hours=1)).strftime("%Y-%m-%d %H"),
    ),
    (
        "I sent mike something titled 'Quick Question' or something - what did i send him?",
        "The email asks a question about the user authentication process (latest feature)."
        " Consider any prediction that mentions the user authentication process as correct.",
    ),
    (
        "What is the title of the events scheduled for 14 days from today?",
        "It's correct so long as the prediction mentions Event 15",
    ),
    (
        "What is the sender of the last email in the thread with ID 1?",
        "sarah@langchain.com",
    ),
    (
        "How many times did I email mike?",
        "Mark either 4 or 2 as correct. There were 4 emails in total across 2 threads.",
    ),
    (
        "what was i supposed to remember for event 25?",
        "The secret code is pikasaurus rex",
    ),
    (
        "When did i get that email titled 'Design Review'?",
        (CURRENT_TIME - timedelta(seconds=16946.851204410043)).isoformat(),
    ),
    (
        "What is the recipient of the email with the oldest timestamp in the thread with ID 5?",
        "jane@langchain.com",
    ),
    (
        "When's the offsite?",
        f"It starts at: {str(get_weekday(4) + timedelta(hours=9))}. Mark as correct if the returned time is any time that day or is a correct relative time (next friday).",
    ),
    (
        "When's my flight for the offsite?",
        f"The flight is {get_weekday(4) + timedelta(hours=5)} to {get_weekday(4) + timedelta(hours=8)}. Accept as correct even if it only mentions the start time.",
    ),
    (
        "What's the name of the hotel for the offsite?",
        "Orange Valley Resort (Also accept Marriott)",
    ),
]


async def format_inputs(inputs: dict):
    return {
        "input": (
            "Evaluate the correctness of the response:\n\n"
            f"<prediction>\n{inputs['prediction']}\n</prediction>\n\n"
            "Compare against the ground truth answer:\n"
            f"<expected>\n{inputs['reference']}\n</expected>"
        )
    }


async def get_score(predictions: list):
    # Output is like: [{'classification': 'incorrect', 'confidence': 1.0}]
    classes = {p["classification"]: p["confidence"] for p in predictions}
    score = None
    if "correct" in classes:
        score = classes["correct"]
    elif "incorrect" in classes:
        score = 1.0 - classes["incorrect"]

    return {"key": "correct-likelihood", "score": score}


_CLASSIFIER = (
    format_inputs
    | create_openai_logprobs_classification_chain(
        ChatOpenAI(model="gpt-4-turbo"),
        {
            "correct": "The prediction contains the correct, expected answer. It may contain other chit chat.",
            "incorrect": "The prediction contradicts the expected answer. Only mark"
            " as incorrect if the prediction contradicts or misses critical factual information from the expected answer.",
        },
    )
    | get_score
).with_config(run_name="CorrectnessClassifier")


@traceable
async def classify(response: str, expected: str):
    run_id = uuid.uuid4()
    result = await _CLASSIFIER.ainvoke(
        {"prediction": response, "reference": expected}, {"run_id": run_id}
    )
    expect.score(
        result["score"], key=result["key"], source_run_id=run_id
    ).to_be_greater_than(0.5)


@pytest.mark.asyncio_cooperative
@pytest.mark.parametrize("question, expected", simple_questions)
async def test_simple_questions(question: str, expected: str) -> str:
    return await check_simple_questions(question, expected)


@unit(output_keys=["expected"])
async def check_simple_questions(question: str, expected: str) -> str:
    error = None
    thread_id = str(uuid.uuid4())
    try:
        result = await assistant_graph.ainvoke(
            {"messages": ("user", question)},
            {"configurable": {"user_id": "vwp@langchain.com", "thread_id": thread_id}},
        )
        response = str(result["messages"][-1].content)
    except Exception as e:
        error = e
        response = f"Error: {repr(e)}"

    if rt := get_current_run_tree():
        rt.outputs = {"result": response}
    expect.embedding_distance(response, expected)
    await classify(response, expected)

    assert error is None, error


@traceable
async def check_event_created(state, tool_kwargs: dict):
    res = await search_calendar_events.ainvoke(tool_kwargs)
    expect(res).against(lambda x: x["count"] > 0)


@traceable
async def check_email_sent(state, tool_kwargs: dict):
    res = await search_emails.ainvoke(tool_kwargs)
    expect(res).against(lambda x: x["count"] > 0)


multistep_questions = [
    (
        [
            "Send an email to joanne @ langchain . com saying we're on for the meeting tomorrow.",
            "Subject should just be 'We're coming to the meeting Tomorrow' - body just 'See you there! - signature should be from VWP",
            "Send whenever you're ready - no rush.",
        ],
        [
            None,
            functools.partial(
                check_email_sent,
                tool_kwargs={
                    "queries": ["joanne"],
                    "start_date": datetime.now() - timedelta(minutes=120),
                },
            ),
        ],
    ),
    (
        [
            "When is the offsite?",
            "What time's my flight?",
            "which airline again?",
            "where are we staying?",
            "create an event with Sachin at the hotel bar 6pm on Friday",
            "send him an email reminder to him",
            "his email is first name @ langchain.com - feel free to send whenever"
            "When are we publicly launching feature the user onboarding feature?",
        ],
        [
            f"It starts at: {get_weekday(4) + timedelta(hours=9)}. Mark as correct if the returned time is any time that day or is a correct relative time (next friday).",
            f"The flight is {get_weekday(4) + timedelta(hours=5)} to {get_weekday(4) + timedelta(hours=8)}. Accept as correct even if it only mentions the start time.",
            "United Airlines",
            "Orange Valley Resort",
            None,
            functools.partial(
                check_event_created,
                tool_kwargs={"queries": ["Sachin"], "start_date": get_weekday(4)},
            ),
            functools.partial(
                check_email_sent,
                tool_kwargs={
                    "queries": ["Sachin"],
                    "start_date": datetime.now() - timedelta(minutes=120),
                },
            ),
            f"Next thursday ({get_weekday(3).strftime('%Y-%m-%d')}) - it's set for 6 PM, though any time that day is considered correct.",
        ],
    ),
]


@pytest.mark.asyncio_cooperative
@pytest.mark.parametrize("questions,expectation_fns", multistep_questions)
async def test_multistep_questions(
    questions: list[str], expectation_fns: list[Callable | str]
) -> str:
    return await check_multistep_questions(questions, expectation_fns)


@unit(output_keys=["expectation_fns"])
async def check_multistep_questions(
    questions: list[str], expectation_fns: list[Callable | str]
) -> str:
    thread_id = str(uuid.uuid4())
    states = []
    responses = []
    config = {
        "configurable": {
            "user_id": "vwp@langchain.com",
            "thread_id": thread_id,
        }
    }
    for question in questions:
        try:
            result = await assistant_graph.ainvoke(
                {"messages": ("user", question)},
                config,
            )
            snapshot = await assistant_graph.aget_state(config)
            if snapshot.next:
                # Just continue - we auto-approve everything here.
                result = await assistant_graph.ainvoke(
                    None,
                    config,
                )
            states.append(states)
            responses.append(result["messages"][-1].content)
        except Exception as e:
            responses.append(f"Error: {repr(e)}")
            states.append(None)

    if rt := get_current_run_tree():
        rt.outputs = {"result": responses}
    errors = []
    with langsmith.trace(name="Evals"):
        for response, expectation_fn in zip(responses, expectation_fns):
            if expectation_fn is None:
                continue
            if isinstance(expectation_fn, str):
                expect.embedding_distance(response, expectation_fn)
                expectation_fn_: Callable = functools.partial(
                    classify, expected=expectation_fn
                )
            else:
                expectation_fn_ = expectation_fn
            try:
                res = expectation_fn_(response)
                # await if it's a coroutine
                if inspect.iscoroutine(res):
                    await res
            except BaseException as e:
                errors.append(e)

    assert not errors, "\n\n".join([repr(e) for e in errors])
