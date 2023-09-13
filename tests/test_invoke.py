from uuid import uuid4

import pytest
from pytest_mock import MockerFixture

from permchain.connection_inmemory import InMemoryPubSubConnection
from permchain.pubsub import PubSub
from permchain.topic import RunnableSubscriber, Topic


def test_invoke_single_process_in_out(mocker: MockerFixture):
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = Topic.IN.subscribe() | add_one | Topic.OUT.publish()

    # Chains can be invoked directly for testing
    assert chain.invoke(2) == 3

    conn = InMemoryPubSubConnection()
    pubsub = PubSub(processes=(chain,), connection=conn)

    # Using in-memory conn internals to make assertions about pubsub
    # If we start with 0 listeners
    assert conn.listeners == {}
    # Then invoke pubsub
    assert pubsub.invoke(2) == [3]
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
    pubsub = PubSub(processes=(chain_one, chain_two), connection=conn)

    # Using in-memory conn internals to make assertions about pubsub
    # If we start with 0 listeners
    assert conn.listeners == {}
    # Then invoke pubsub
    assert pubsub.invoke(2) == [4]
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
    pubsub_one = PubSub(processes=(chain_one,), connection=conn)
    pubsub_two = PubSub(processes=(chain_two,), connection=conn)

    # Using in-memory conn internals to make assertions about pubsub
    # If we start with 0 listeners
    assert conn.listeners == {}
    # Then invoke both pubsubs, as a group
    # The second picks up where the first left off
    state_id = uuid4()

    # invoke() step 1
    assert pubsub_one.invoke(2, {"state_id": state_id}) == []

    # listeners are still cleared, even though state is preserved
    assert conn.listeners == {}
    # The log contains all messages published to all topics, in order
    assert [{**m, "started_at": None} for m in conn.peek(state_id)] == [
        {"message": 2, "topic_name": "__in__", "started_at": None},
        {"message": 3, "topic_name": "one", "started_at": None},
    ]
    # IN, OUT, one
    assert len(conn.topics) == 3
    topic_one_full_name = conn.full_topic_name(state_id, topic_one.name)
    # the actual message publishd by chain_one, and a sentinel "end" value
    assert conn.topics[topic_one_full_name].qsize() == 2

    # invoke() step 2
    # this picks up where the first left off, and produces same result as
    # `test_invoke_two_processes_in_out`
    assert pubsub_two.invoke(None, {"state_id": state_id}) == [4]

    # listeners are still cleared, even though state is preserved
    assert conn.listeners == {}
    # The log contains all messages published to all topics, in order
    assert [{**m, "started_at": None} for m in conn.peek(state_id)] == [
        {"message": 2, "topic_name": "__in__", "started_at": None},
        {"message": 3, "topic_name": "one", "started_at": None},
        {"message": None, "topic_name": "__in__", "started_at": None},
        {"message": 4, "topic_name": "__out__", "started_at": None},
    ]
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
            # Contains sentinel "end" value
            assert queue.qsize() == 1


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
    assert pubsub.invoke(2) == [2 + test_size]
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
    # We get two equal results as the two chains do the same thing
    assert pubsub.invoke(2) == [3, 3]

    # After invoke returns the listeners were cleaned up
    assert conn.listeners == {}


def test_invoke_two_processes_two_in_reduce_two_out(mocker: MockerFixture):
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    add_10_each = mocker.Mock(side_effect=lambda x: [y + 10 for y in x])
    topic_one = Topic("one")
    topic_two = Topic("two")
    chain_one = Topic.IN.subscribe() | add_one | topic_one.publish()
    chain_two = topic_one.subscribe() | add_one | topic_two.publish()
    chain_three = Topic.IN.subscribe() | add_one | topic_two.publish()
    chain_four = topic_two.reduce() | add_10_each | Topic.OUT.publish()

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
    assert pubsub.invoke(2) == [[13, 14]]

    # After invoke returns the listeners were cleaned up
    assert conn.listeners == {}


def test_invoke_reduce_then_subscribe(mocker: MockerFixture):
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    add_10_each = mocker.Mock(side_effect=lambda x: [y + 10 for y in x])

    topic_one = Topic("one")
    topic_two = Topic("two")

    chain_one = Topic.IN.subscribe() | add_10_each | topic_one.publish_each()
    chain_two = topic_one.reduce() | sum | topic_two.publish()
    chain_three = topic_two.subscribe() | add_one | Topic.OUT.publish()

    # Chains can be invoked directly for testing
    assert chain_two.invoke([2, 3]) == 5
    assert chain_three.invoke(5) == 6

    state_id = uuid4()
    conn = InMemoryPubSubConnection(clear_on_disconnect=False)
    pubsub = PubSub((chain_one, chain_two, chain_three), connection=conn)

    # Using in-memory conn internals to make assertions about pubsub
    # If we start with 0 listeners
    assert conn.listeners == {}

    # Then invoke pubsub
    # We get a single array result as chain_four waits for all publishers to finish
    # before operating on all elements published to topic_two as an array
    assert pubsub.invoke([2, 3], {"state_id": state_id}) == [26]
    assert [{**m, "started_at": None} for m in conn.peek(state_id)] == [
        {"message": [2, 3], "topic_name": "__in__", "started_at": None},
        {"message": 12, "topic_name": "one", "started_at": None},
        {"message": 13, "topic_name": "one", "started_at": None},
        {"message": 25, "topic_name": "two", "started_at": None},
        {"message": 26, "topic_name": "__out__", "started_at": None},
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
    # pubsub didn't stop executing after getting the first return value
    # the values arrive in the order they are produced
    assert pubsub.invoke(2) == [3, 4]

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
    assert pubsub.invoke(2) == []

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
    assert pubsub.invoke(2) == []

    # After invoke returns the listeners were cleaned up
    assert conn.listeners == {}


@pytest.mark.skip("TODO")
def test_invoke_two_processes_simple_cycle(mocker: MockerFixture):
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
