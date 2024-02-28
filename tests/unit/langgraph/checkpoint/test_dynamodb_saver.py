import boto3
from moto import mock_aws
import pytest
import os

import operator
from langchain_core.messages import BaseMessage
from typing import Annotated, Sequence, TypedDict

from langgraph.graph import StateGraph, END

from langchain.llms.fake import FakeListLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser

# SUT
from langgraph.checkpoint.dynamodb import DynamoDBSaver


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


@pytest.fixture(scope="function")
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture(scope="function")
def dynamodb(aws_credentials):
    with mock_aws():
        yield boto3.resource("dynamodb", region_name="us-east-1")


@pytest.fixture(scope="function")
def table_name():
    return "fake_table_name"


@pytest.fixture(scope="function")
def table(dynamodb, table_name):
    dynamodb.create_table(TableName=table_name,
                          KeySchema=[{'AttributeName': 'thread_id', 'KeyType': 'HASH'}],
                          BillingMode="PAY_PER_REQUEST",
                          AttributeDefinitions=[{'AttributeName': 'thread_id', 'AttributeType': 'S'}])

    return dynamodb.Table(table_name)


@pytest.fixture(scope="function")
def message_from_node2():
    return "message_from_node2"


@pytest.fixture(scope="function")
def workflow(message_from_node2):
    workflow = StateGraph(AgentState)

    node1 = (
            PromptTemplate.from_template("Mock template with {messages}") |
            FakeListLLM(responses=['{"next": "node2"}']) |
            JsonOutputParser())

    node2 = (
            PromptTemplate.from_template("Another mock template") |
            FakeListLLM(responses=[f'{{"messages": ["{message_from_node2}"]}}']) |
            JsonOutputParser())

    workflow.add_node("node1", node1)
    workflow.add_node("node2", node2)

    workflow.set_entry_point("node1")
    workflow.add_edge("node1", "node2")
    workflow.add_edge("node2", END)

    return workflow


def test_repository_can_add_creators(table, table_name, message_from_node2, workflow):
    a_thread_id = "uniqueId"
    a_next_node = "node2"
    a_new_message = "hello!"

    memory = DynamoDBSaver.from_table(table_name)
    graph = workflow.compile(checkpointer=memory)

    new_state = graph.invoke({"messages": [a_new_message]}, {"configurable": {"thread_id": a_thread_id}})
    print(new_state, flush=True)

    assert new_state == {"messages": [a_new_message, message_from_node2], "next": a_next_node}

    memory_item = table.get_item(Key={"thread_id": a_thread_id})

    assert "Item" in memory_item

    assert memory_item["Item"]["channel_values"]["next"] == a_next_node
    assert memory_item["Item"]["channel_values"]["messages"] == [a_new_message, message_from_node2]
