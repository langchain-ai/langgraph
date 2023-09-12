import pytest
from pytest_mock import MockerFixture

from langchain.callbacks.manager import trace_as_chain_group
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
    with trace_as_chain_group("PubSubGroup") as cm:
        # invoke() step 1
        assert pubsub_one.invoke(2, {"callbacks": cm}) == []

        # listeners are still cleared, even though state is preserved
        assert conn.listeners == {}
        # The log contains all messages published to all topics, in order
        assert [{**m, "started_at": None} for m in conn.peek(cm.parent_run_id)] == [
            {"message": 2, "topic_name": "__in__", "started_at": None},
            {"message": 3, "topic_name": "one", "started_at": None},
        ]
        # IN, OUT, one
        assert len(conn.topics) == 3
        topic_one_full_name = conn.full_topic_name(cm.parent_run_id, topic_one.name)
        # the actual message publishd by chain_one, and a sentinel "end" value
        assert conn.topics[topic_one_full_name].qsize() == 2

        # invoke() step 2
        # this picks up where the first left off, and produces same result as
        # `test_invoke_two_processes_in_out`
        assert pubsub_two.invoke(None, {"callbacks": cm}) == [4]

        # listeners are still cleared, even though state is preserved
        assert conn.listeners == {}
        # The log contains all messages published to all topics, in order
        assert [{**m, "started_at": None} for m in conn.peek(cm.parent_run_id)] == [
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
