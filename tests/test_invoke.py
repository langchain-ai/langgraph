from typing import Iterator
from uuid import uuid4

import pytest
from pytest_mock import MockerFixture

from permchain.connection import PubSubMessage
from permchain.connection_inmemory import InMemoryPubSubConnection
from permchain.pubsub import PubSub
from permchain.topic import RunnableSubscriber, Topic


def clean_log(
    logs: Iterator[PubSubMessage], correlation_id: bool | None = None
) -> list[PubSubMessage]:
    if correlation_id is False:
        return [{**m, "published_at": None, "correlation_id": None} for m in logs]
    else:
        return [{**m, "published_at": None} for m in logs]


def test_invoke_single_process_in_out(mocker: MockerFixture):
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = Topic.IN.subscribe() | add_one | Topic.OUT.publish()

    # Chains can be invoked directly for testing
    assert chain.invoke(2) == 3

    conn = InMemoryPubSubConnection()
    pubsub = PubSub(chain, connection=conn)

    # Using in-memory conn internals to make assertions about pubsub
    # If we start with 0 listeners
    assert conn.listeners == {}
    # Then invoke pubsub
    assert pubsub.invoke(2) == 3
    # After invoke returns the listeners were cleaned up
    assert conn.listeners == {}


def test_invoke_two_processes_in_out(mocker: MockerFixture):
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    topic_one = Topic("one")
    chain_one = Topic.IN.subscribe() | add_one | topic_one.publish()
    chain_two = topic_one.subscribe() | add_one | Topic.OUT.publish()

    # Chains can be invoked directly for testing
    assert chain_one.invoke(2) == 3
    assert chain_two.invoke(2) == 3

    conn = InMemoryPubSubConnection()
    pubsub = PubSub(chain_one, chain_two, connection=conn)

    # Using in-memory conn internals to make assertions about pubsub
    # If we start with 0 listeners
    assert conn.listeners == {}
    # Then invoke pubsub
    assert pubsub.invoke(2) == 4
    # After invoke returns the listeners were cleaned up
    assert conn.listeners == {}


def test_invoke_two_processes_in_out_interrupt(mocker: MockerFixture):
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    topic_one = Topic("one")
    chain_one = Topic.IN.subscribe() | add_one | topic_one.publish()
    chain_two = topic_one.subscribe() | add_one | Topic.OUT.publish()

    # Chains can be invoked directly for testing
    assert chain_one.invoke(2) == 3
    assert chain_two.invoke(2) == 3

    conn = InMemoryPubSubConnection(clear_on_disconnect=False)
    pubsub_one = PubSub(chain_one, connection=conn)
    pubsub_two = PubSub(chain_two, connection=conn)

    # Using in-memory conn internals to make assertions about pubsub
    # If we start with 0 listeners
    assert conn.listeners == {}
    # Then invoke both pubsubs, as a group
    # The second picks up where the first left off
    correlation_id = uuid4()

    # invoke() step 1
    assert clean_log(pubsub_one.stream(2, {"correlation_id": correlation_id})) == [
        {
            "value": 2,
            "topic": "__in__",
            "correlation_id": str(correlation_id),
            "published_at": None,
        },
        {
            "value": 3,
            "topic": "one",
            "correlation_id": str(correlation_id),
            "published_at": None,
        },
    ]

    # IN, one
    assert len(conn.topics) == 2
    topic_one_full_name = conn.full_name(correlation_id, topic_one.name)
    # the actual message publishd by chain_one, and a sentinel "end" value
    assert conn.topics[topic_one_full_name].qsize() == 2

    # invoke() step 2
    # this picks up where the first left off, and produces same result as
    # `test_invoke_two_processes_in_out`
    assert clean_log(pubsub_two.stream(None, {"correlation_id": correlation_id})) == [
        {
            "value": None,
            "topic": "__in__",
            "correlation_id": str(correlation_id),
            "published_at": None,
        },
        {
            "value": 4,
            "topic": "__out__",
            "correlation_id": str(correlation_id),
            "published_at": None,
        },
    ]
    # listeners are still cleared, even though state is preserved
    assert conn.listeners == {}
    # IN, OUT, one
    assert len(conn.topics) == 3

    for topic_name, queue in conn.topics.items():
        if topic_name.endswith("IN"):
            # Contains two sentinel "end" values, and None
            # passed in as input to chain_two, which doesn't subscribe to it
            assert queue.qsize() == 3
        if topic_name.endswith("OUT"):
            # Empty because this was consumed by invoke()
            assert queue.qsize() == 0
        if topic_name.endswith("one"):
            # Contains 2 sentinel "end" values
            assert queue.qsize() == 2


def test_invoke_many_processes_in_out(mocker: MockerFixture):
    test_size = 100

    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    topics: list[Topic] = [Topic("zero")]
    chains: list[RunnableSubscriber] = [
        Topic.IN.subscribe() | add_one | topics[0].publish()
    ]
    for i in range(test_size - 2):
        topics.append(Topic(str(i)))
        chains.append(topics[-2].subscribe() | add_one | topics[-1].publish())
    chains.append(topics[-1].subscribe() | add_one | Topic.OUT.publish())

    # Chains can be invoked directly for testing
    for chain in chains:
        assert chain.invoke(2) == 3

    conn = InMemoryPubSubConnection()
    pubsub = PubSub(processes=chains, connection=conn)

    # Using in-memory conn internals to make assertions about pubsub
    # If we start with 0 listeners
    assert conn.listeners == {}
    # Then invoke pubsub
    assert pubsub.invoke(2) == 2 + test_size
    # After invoke returns the listeners were cleaned up
    assert conn.listeners == {}


def test_invoke_two_processes_two_in_two_out(mocker: MockerFixture):
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain_one = Topic.IN.subscribe() | add_one | Topic.OUT.publish()
    chain_two = Topic.IN.subscribe() | add_one | Topic.OUT.publish()

    # Chains can be invoked directly for testing
    assert chain_one.invoke(2) == 3
    assert chain_two.invoke(2) == 3

    conn = InMemoryPubSubConnection()
    pubsub = PubSub(processes=(chain_one, chain_two), connection=conn)

    # Using in-memory conn internals to make assertions about pubsub
    # If we start with 0 listeners
    assert conn.listeners == {}

    # Then invoke pubsub
    # We get only one of the two return values, as computation is closed
    # as soon as we publish to OUT for the first time
    assert pubsub.invoke(2) == 3

    # After invoke returns the listeners were cleaned up
    assert conn.listeners == {}


def test_invoke_two_processes_two_in_join_two_out(mocker: MockerFixture):
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    add_10_each = mocker.Mock(side_effect=lambda x: sorted(y + 10 for y in x))
    topic_one = Topic("one")
    topic_two = Topic("two")
    chain_one = Topic.IN.subscribe() | add_one | topic_one.publish()
    chain_two = topic_one.subscribe() | add_one | topic_two.publish()
    chain_three = Topic.IN.subscribe() | add_one | topic_two.publish()
    chain_four = topic_two.join() | add_10_each | Topic.OUT.publish()

    # Chains can be invoked directly for testing
    assert chain_one.invoke(2) == 3
    assert chain_four.invoke([2, 3]) == [12, 13]

    conn = InMemoryPubSubConnection()
    pubsub = PubSub((chain_one, chain_two, chain_three, chain_four), connection=conn)

    # Using in-memory conn internals to make assertions about pubsub
    # If we start with 0 listeners
    assert conn.listeners == {}

    # Then invoke pubsub
    # We get a single array result as chain_four waits for all publishers to finish
    # before operating on all elements published to topic_two as an array
    assert pubsub.invoke(2) == [13, 14]

    # After invoke returns the listeners were cleaned up
    assert conn.listeners == {}


def test_invoke_join_then_subscribe(mocker: MockerFixture):
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    add_10_each = mocker.Mock(side_effect=lambda x: [y + 10 for y in x])

    topic_one = Topic("one")
    topic_two = Topic("two")

    chain_one = Topic.IN.subscribe() | add_10_each | topic_one.publish_each()
    chain_two = topic_one.join() | sum | topic_two.publish()
    chain_three = topic_two.subscribe() | add_one | Topic.OUT.publish()

    # Chains can be invoked directly for testing
    assert chain_two.invoke([2, 3]) == 5
    assert chain_three.invoke(5) == 6

    correlation_id = uuid4()
    conn = InMemoryPubSubConnection(clear_on_disconnect=False)
    pubsub = PubSub((chain_one, chain_two, chain_three), connection=conn)

    # Using in-memory conn internals to make assertions about pubsub
    # If we start with 0 listeners
    assert conn.listeners == {}

    # Then invoke pubsub
    # We get a single array result as chain_four waits for all publishers to finish
    # before operating on all elements published to topic_two as an array
    assert clean_log(pubsub.stream([2, 3], {"correlation_id": correlation_id})) == [
        {
            "value": [2, 3],
            "topic": "__in__",
            "correlation_id": str(correlation_id),
            "published_at": None,
        },
        {
            "value": 12,
            "topic": "one",
            "correlation_id": str(correlation_id),
            "published_at": None,
        },
        {
            "value": 13,
            "topic": "one",
            "correlation_id": str(correlation_id),
            "published_at": None,
        },
        {
            "value": 25,
            "topic": "two",
            "correlation_id": str(correlation_id),
            "published_at": None,
        },
        {
            "value": 26,
            "topic": "__out__",
            "correlation_id": str(correlation_id),
            "published_at": None,
        },
    ]

    # After invoke returns the listeners were cleaned up
    assert conn.listeners == {}


def test_invoke_join_then_call_other_pubsub(mocker: MockerFixture):
    conn = InMemoryPubSubConnection(clear_on_disconnect=False)
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    inner_pubsub = PubSub(
        (Topic.IN.subscribe() | add_one | Topic.OUT.publish(),), connection=conn
    )

    add_10_each = mocker.Mock(side_effect=lambda x: [y + 10 for y in x])

    topic_one = Topic("one")
    topic_two = Topic("two")

    chain_one = Topic.IN.subscribe() | add_10_each | topic_one.publish_each()
    chain_two = topic_one.join() | inner_pubsub.map() | sorted | topic_two.publish()
    chain_three = topic_two.subscribe() | sum | Topic.OUT.publish()

    correlation_id = uuid4()
    pubsub = PubSub((chain_one, chain_two, chain_three), connection=conn)

    # Using in-memory conn internals to make assertions about pubsub
    # If we start with 0 listeners
    assert conn.listeners == {}

    # Then invoke pubsub
    assert clean_log(pubsub.stream([2, 3], {"correlation_id": correlation_id})) == [
        {
            "value": [2, 3],
            "topic": "__in__",
            "correlation_id": str(correlation_id),
            "published_at": None,
        },
        {
            "value": 12,
            "topic": "one",
            "correlation_id": str(correlation_id),
            "published_at": None,
        },
        {
            "value": 13,
            "topic": "one",
            "correlation_id": str(correlation_id),
            "published_at": None,
        },
        {
            "value": [13, 14],
            "topic": "two",
            "correlation_id": str(correlation_id),
            "published_at": None,
        },
        {
            "value": 27,
            "topic": "__out__",
            "correlation_id": str(correlation_id),
            "published_at": None,
        },
    ]

    # After invoke returns the listeners were cleaned up
    assert conn.listeners == {}


def test_invoke_two_processes_one_in_two_out(mocker: MockerFixture):
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    topic_one = Topic("one")
    # Topic.publish() is passthrough so we can publish to multiple topics in sequence
    chain_one = (
        Topic.IN.subscribe() | add_one | Topic.OUT.publish() | topic_one.publish()
    )
    chain_two = topic_one.subscribe() | add_one | Topic.OUT.publish()

    # Chains can be invoked directly for testing
    assert chain_one.invoke(2) == 3
    assert chain_two.invoke(2) == 3

    conn = InMemoryPubSubConnection()
    pubsub = PubSub(processes=(chain_one, chain_two), connection=conn)

    # Using in-memory conn internals to make assertions about pubsub
    # If we start with 0 listeners
    assert conn.listeners == {}

    # Then invoke pubsub
    # pubsub stopped executing after publishing to OUT, so only one value is returned
    assert clean_log(pubsub.stream(2), correlation_id=False) == [
        {
            "value": 2,
            "topic": "__in__",
            "correlation_id": None,
            "published_at": None,
        },
        {
            "value": 3,
            "topic": "__out__",
            "correlation_id": None,
            "published_at": None,
        },
    ]

    # After invoke returns the listeners were cleaned up
    assert conn.listeners == {}


def test_invoke_two_processes_no_out(mocker: MockerFixture):
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    topic_one = Topic("one")
    chain_one = Topic.IN.subscribe() | add_one | topic_one.publish()
    chain_two = topic_one.subscribe() | add_one

    # Chains can be invoked directly for testing
    assert chain_one.invoke(2) == 3
    assert chain_two.invoke(2) == 3

    conn = InMemoryPubSubConnection()
    pubsub = PubSub(processes=(chain_one, chain_two), connection=conn)

    # Using in-memory conn internals to make assertions about pubsub
    # If we start with 0 listeners
    assert conn.listeners == {}

    # Then invoke pubsub
    # It finishes executing (once no more messages being published)
    # but returns nothing, as nothing was published to OUT topic
    assert pubsub.invoke(2) is None

    # After invoke returns the listeners were cleaned up
    assert conn.listeners == {}


def test_invoke_two_processes_no_in(mocker: MockerFixture):
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    topic_one = Topic("one")
    chain_one = topic_one.subscribe() | add_one | Topic.OUT.publish()
    chain_two = topic_one.subscribe() | add_one | Topic.OUT.publish()

    # Chains can be invoked directly for testing
    assert chain_one.invoke(2) == 3
    assert chain_two.invoke(2) == 3

    conn = InMemoryPubSubConnection()
    pubsub = PubSub(processes=(chain_one, chain_two), connection=conn)

    # Using in-memory conn internals to make assertions about pubsub
    # If we start with 0 listeners
    assert conn.listeners == {}

    # Then invoke pubsub
    # It returns without any output as there is nothing to run
    assert pubsub.invoke(2) is None

    # After invoke returns the listeners were cleaned up
    assert conn.listeners == {}


@pytest.mark.skip("TODO")
def test_invoke_two_processes_simple_cycle(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    topic_one = Topic("one")
    chain_one = Topic.IN.subscribe() | add_one | topic_one.publish()
    chain_two = topic_one.subscribe() | add_one | topic_one.publish()

    # Chains can be invoked directly for testing
    assert chain_one.invoke(2) == 3
    assert chain_two.invoke(2) == 3

    conn = InMemoryPubSubConnection()
    pubsub = PubSub(processes=(chain_one, chain_two), connection=conn)

    # Using in-memory conn internals to make assertions about pubsub
    # If we start with 0 listeners
    assert conn.listeners == {}
    # Then invoke pubsub
    with pytest.raises(RecursionError):
        pubsub.invoke(2)
    # After invoke returns the listeners were cleaned up
    for key in conn.listeners:
        assert not conn.listeners[key]
