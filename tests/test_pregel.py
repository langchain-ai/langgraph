import time
from uuid import uuid4

import pytest
from pytest_mock import MockerFixture
from langchain.schema.runnable import RunnablePassthrough

from permchain import Pregel, channels
from permchain.pregel import PregelInvoke


def test_invoke_single_process_in_out(mocker: MockerFixture):
    input = channels.LastValue[int]("input")
    output = channels.LastValue[int]("output")

    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = Pregel.subscribe_to(input) | add_one | Pregel.send_to(output)

    pubsub = Pregel(chain, input=input, output=output)

    # Then invoke pubsub
    assert pubsub.invoke(2) == 3


def test_invoke_two_processes_in_out(mocker: MockerFixture):
    input = channels.LastValue[int]("input")
    output = channels.LastValue[int]("output")
    inbox = channels.Inbox[int]("inbox")

    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain_one = Pregel.subscribe_to(input) | add_one | Pregel.send_to(inbox)
    chain_two = Pregel.subscribe_to_each(inbox) | add_one | Pregel.send_to(output)

    pubsub = Pregel(chain_one, chain_two, input=input, output=output)

    # Then invoke pubsub
    assert pubsub.invoke(2) == 4


def test_batch_two_processes_in_out(mocker: MockerFixture):
    def add_one_with_delay(inp: int) -> int:
        time.sleep(inp / 10)
        return inp + 1

    input = channels.LastValue[int]("input")
    output = channels.LastValue[int]("output")
    one = channels.LastValue[int]("one")
    chain_one = Pregel.subscribe_to(input) | add_one_with_delay | Pregel.send_to(one)
    chain_two = Pregel.subscribe_to(one) | add_one_with_delay | Pregel.send_to(output)

    pubsub = Pregel(chain_one, chain_two, input=input, output=output)

    # Then invoke pubsub
    assert pubsub.batch([3, 2, 1, 3, 5]) == [5, 4, 3, 5, 7]


def test_invoke_many_processes_in_out(mocker: MockerFixture):
    test_size = 100

    input = channels.LastValue[int]("input")
    output = channels.LastValue[int]("output")
    topics: list[channels.Channel] = [channels.LastValue[int]("zero")]

    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chains: list[PregelInvoke] = [
        Pregel.subscribe_to(input) | add_one | Pregel.send_to(topics[0])
    ]
    for i in range(test_size - 2):
        topics.append(channels.LastValue[int](str(i)))
        chains.append(
            Pregel.subscribe_to(topics[-2]) | add_one | Pregel.send_to(topics[-1])
        )
    chains.append(Pregel.subscribe_to(topics[-1]) | add_one | Pregel.send_to(output))

    pubsub = Pregel(*chains, input=input, output=output)

    for _ in range(10):
        # Then invoke pubsub
        assert pubsub.invoke(2, {"recursion_limit": test_size}) == 2 + test_size


def test_batch_many_processes_in_out(mocker: MockerFixture):
    test_size = 100

    input = channels.LastValue[int]("input")
    output = channels.LastValue[int]("output")
    topics: list[channels.Channel] = [channels.LastValue[int]("zero")]

    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chains: list[PregelInvoke] = [
        Pregel.subscribe_to(input) | add_one | Pregel.send_to(topics[0])
    ]
    for i in range(test_size - 2):
        topics.append(channels.LastValue[int](str(i)))
        chains.append(
            Pregel.subscribe_to(topics[-2]) | add_one | Pregel.send_to(topics[-1])
        )
    chains.append(Pregel.subscribe_to(topics[-1]) | add_one | Pregel.send_to(output))

    pubsub = Pregel(*chains, input=input, output=output)

    for _ in range(10):
        # Then invoke pubsub
        assert pubsub.batch([2, 1, 3, 4, 5], {"recursion_limit": test_size}) == [
            2 + test_size,
            1 + test_size,
            3 + test_size,
            4 + test_size,
            5 + test_size,
        ]


def test_invoke_two_processes_two_in_two_out_invalid(mocker: MockerFixture):
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    input = channels.LastValue[int]("input")
    output = channels.LastValue[int]("output")

    chain_one = Pregel.subscribe_to(input) | add_one | Pregel.send_to(output)
    chain_two = Pregel.subscribe_to(input) | add_one | Pregel.send_to(output)

    pubsub = Pregel(chain_one, chain_two, input=input, output=output)

    with pytest.raises(channels.InvalidUpdateError):
        # LastValue channels can only be updated once per iteration
        pubsub.invoke(2)


def test_invoke_two_processes_two_in_two_out_valid(mocker: MockerFixture):
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    input = channels.LastValue[int]("input")
    output = channels.Inbox[int]("output")

    chain_one = Pregel.subscribe_to(input) | add_one | Pregel.send_to(output)
    chain_two = Pregel.subscribe_to(input) | add_one | Pregel.send_to(output)

    pubsub = Pregel(chain_one, chain_two, input=input, output=output)

    # An Inbox channel accumulates updates into a sequence
    assert pubsub.invoke(2) == (3, 3)


def test_invoke_two_processes_two_in_join_two_out(mocker: MockerFixture):
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    add_10_each = mocker.Mock(side_effect=lambda x: sorted(y + 10 for y in x))

    input = channels.LastValue[int]("input")
    output = channels.LastValue[int]("output")
    inbox = channels.Inbox[int]("inbox")

    chain_one = Pregel.subscribe_to(input) | add_one | Pregel.send_to(inbox)
    chain_three = Pregel.subscribe_to(input) | add_one | Pregel.send_to(inbox)
    chain_four = Pregel.subscribe_to(inbox) | add_10_each | Pregel.send_to(output)

    pubsub = Pregel(chain_one, chain_three, chain_four, input=input, output=output)

    # Then invoke pubsub
    # We get a single array result as chain_four waits for all publishers to finish
    # before operating on all elements published to topic_two as an array
    for _ in range(100):
        assert pubsub.invoke(2) == [13, 13]


def test_invoke_join_then_call_other_pubsub(mocker: MockerFixture):
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    add_10_each = mocker.Mock(side_effect=lambda x: [y + 10 for y in x])

    input = channels.LastValue[int]("input")
    output = channels.LastValue[int]("output")

    inner_pubsub = Pregel(
        Pregel.subscribe_to(input) | add_one | Pregel.send_to(output),
        input=input,
        output=output,
    )

    inbox_one = channels.Inbox[int]("inbox_one")
    outbox_one = channels.LastValue[int]("outbox_one")

    chain_one = (
        Pregel.subscribe_to(input) | add_10_each | Pregel.send_to(inbox_one).map()
    )
    chain_two = (
        Pregel.subscribe_to(inbox_one)
        | inner_pubsub.map()
        | sorted
        | Pregel.send_to(outbox_one)
    )
    chain_three = Pregel.subscribe_to(outbox_one) | sum | Pregel.send_to(output)

    pubsub = Pregel(chain_one, chain_two, chain_three, input=input, output=output)

    # Then invoke pubsub
    for _ in range(10):
        assert pubsub.invoke([2, 3]) == 27


def test_invoke_two_processes_one_in_two_out(mocker: MockerFixture):
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    input = channels.LastValue[int]("input")
    output = channels.LastValue[int]("output")
    between = channels.LastValue[int]("between")

    chain_one = (
        Pregel.subscribe_to(input)
        | add_one
        | Pregel.send_to(
            {output: RunnablePassthrough(), between: RunnablePassthrough()}
        )
    )
    chain_two = Pregel.subscribe_to(between) | add_one | Pregel.send_to(output)

    pubsub = Pregel(chain_one, chain_two, input=input, output=output)

    # Then invoke pubsub
    assert [c for c in pubsub.stream(2)] == [3, 4]


def test_invoke_two_processes_no_out(mocker: MockerFixture):
    input = channels.LastValue[int]("input")
    output = channels.LastValue[int]("output")
    between = channels.LastValue[int]("between")

    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain_one = Pregel.subscribe_to(input) | add_one | Pregel.send_to(between)
    chain_two = Pregel.subscribe_to(between) | add_one

    pubsub = Pregel(chain_one, chain_two, input=input, output=output)

    # Then invoke pubsub
    # It finishes executing (once no more messages being published)
    # but returns nothing, as nothing was published to OUT topic
    assert pubsub.invoke(2) is None


def test_invoke_two_processes_no_in(mocker: MockerFixture):
    input = channels.LastValue[int]("input")
    output = channels.LastValue[int]("output")
    between = channels.LastValue[int]("between")

    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain_one = Pregel.subscribe_to(between) | add_one | Pregel.send_to(output)
    chain_two = Pregel.subscribe_to(between) | add_one

    pubsub = Pregel(chain_one, chain_two, input=input, output=output)

    with pytest.raises(ValueError):
        assert pubsub.invoke(2) is None
