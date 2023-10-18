import asyncio

import pytest
from langchain.schema.runnable import RunnablePassthrough
from pytest_mock import MockerFixture

from permchain import Pregel, channels
from permchain.pregel import PregelInvoke


async def test_invoke_single_process_in_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = Pregel.subscribe_to("input") | add_one | Pregel.send_to("output")

    app = Pregel(
        (chain,),
        channels={
            "input": channels.LastValue(int),
            "output": channels.LastValue(int),
        },
        input="input",
        output="output",
    )

    assert await app.ainvoke(2) == 3


async def test_invoke_two_processes_in_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain_one = Pregel.subscribe_to("input") | add_one | Pregel.send_to("inbox")
    chain_two = Pregel.subscribe_to_each("inbox") | add_one | Pregel.send_to("output")

    app = Pregel(
        [chain_one, chain_two],
        channels={
            "input": channels.LastValue(int),
            "output": channels.LastValue(int),
            "inbox": channels.Inbox(int),
        },
        input="input",
        output="output",
    )

    assert await app.ainvoke(2) == 4


async def test_batch_two_processes_in_out() -> None:
    async def add_one_with_delay(inp: int) -> int:
        await asyncio.sleep(inp / 10)
        return inp + 1

    chain_one = (
        Pregel.subscribe_to("input") | add_one_with_delay | Pregel.send_to("one")
    )
    chain_two = (
        Pregel.subscribe_to("one") | add_one_with_delay | Pregel.send_to("output")
    )

    app = Pregel(
        chain_one,
        chain_two,
        channels={
            "input": channels.LastValue(int),
            "output": channels.LastValue(int),
            "one": channels.LastValue(int),
        },
        input="input",
        output="output",
    )

    assert await app.abatch([3, 2, 1, 3, 5]) == [5, 4, 3, 5, 7]


async def test_invoke_many_processes_in_out(mocker: MockerFixture) -> None:
    test_size = 100
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    chans = {
        "input": channels.LastValue(int),
        "output": channels.LastValue(int),
        "-1": channels.LastValue(int),
    }
    chains: list[PregelInvoke] = [
        Pregel.subscribe_to("input") | add_one | Pregel.send_to("-1")
    ]
    for i in range(test_size - 2):
        chans[str(i)] = channels.LastValue(int)
        chains.append(
            Pregel.subscribe_to(str(i - 1)) | add_one | Pregel.send_to(str(i))
        )
    chains.append(Pregel.subscribe_to(str(i)) | add_one | Pregel.send_to("output"))

    app = Pregel(*chains, channels=chans, input="input", output="output")

    # No state is left over from previous invocations
    for _ in range(10):
        assert await app.ainvoke(2, {"recursion_limit": test_size}) == 2 + test_size

    # Concurrent invocations do not interfere with each other
    assert await asyncio.gather(
        *(app.ainvoke(2, {"recursion_limit": test_size}) for _ in range(10))
    ) == [2 + test_size for _ in range(10)]


async def test_batch_many_processes_in_out(mocker: MockerFixture) -> None:
    test_size = 100
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    chans = {
        "input": channels.LastValue(int),
        "output": channels.LastValue(int),
        "-1": channels.LastValue(int),
    }
    chains: list[PregelInvoke] = [
        Pregel.subscribe_to("input") | add_one | Pregel.send_to("-1")
    ]
    for i in range(test_size - 2):
        chans[str(i)] = channels.LastValue(int)
        chains.append(
            Pregel.subscribe_to(str(i - 1)) | add_one | Pregel.send_to(str(i))
        )
    chains.append(Pregel.subscribe_to(str(i)) | add_one | Pregel.send_to("output"))

    app = Pregel(*chains, channels=chans, input="input", output="output")

    # No state is left over from previous invocations
    for _ in range(10):
        # Then invoke pubsub
        assert await app.abatch([2, 1, 3, 4, 5], {"recursion_limit": test_size}) == [
            2 + test_size,
            1 + test_size,
            3 + test_size,
            4 + test_size,
            5 + test_size,
        ]

    # Concurrent invocations do not interfere with each other
    assert await asyncio.gather(
        *(
            app.abatch([2, 1, 3, 4, 5], {"recursion_limit": test_size})
            for _ in range(10)
        )
    ) == [
        [2 + test_size, 1 + test_size, 3 + test_size, 4 + test_size, 5 + test_size]
        for _ in range(10)
    ]


async def test_invoke_two_processes_two_in_two_out_invalid(
    mocker: MockerFixture,
) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    chain_one = Pregel.subscribe_to("input") | add_one | Pregel.send_to("output")
    chain_two = Pregel.subscribe_to("input") | add_one | Pregel.send_to("output")

    app = Pregel(
        chain_one,
        chain_two,
        channels={
            "input": channels.LastValue(int),
            "output": channels.LastValue(int),
        },
        input="input",
        output="output",
    )

    with pytest.raises(channels.InvalidUpdateError):
        # LastValue channels can only be updated once per iteration
        await app.ainvoke(2)


async def test_invoke_two_processes_two_in_two_out_valid(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    chain_one = Pregel.subscribe_to("input") | add_one | Pregel.send_to("output")
    chain_two = Pregel.subscribe_to("input") | add_one | Pregel.send_to("output")

    app = Pregel(
        chain_one,
        chain_two,
        channels={
            "input": channels.LastValue(int),
            "output": channels.Inbox(int),
        },
        input="input",
        output="output",
    )

    # An Inbox channel accumulates updates into a sequence
    assert await app.ainvoke(2) == (3, 3)


async def test_invoke_two_processes_two_in_join_two_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    add_10_each = mocker.Mock(side_effect=lambda x: sorted(y + 10 for y in x))

    chain_one = Pregel.subscribe_to("input") | add_one | Pregel.send_to("inbox")
    chain_three = Pregel.subscribe_to("input") | add_one | Pregel.send_to("inbox")
    chain_four = Pregel.subscribe_to("inbox") | add_10_each | Pregel.send_to("output")

    app = Pregel(
        chain_one,
        chain_three,
        chain_four,
        channels={
            "input": channels.LastValue(int),
            "output": channels.LastValue(int),
            "inbox": channels.Inbox(int),
        },
        input="input",
        output="output",
    )

    # Then invoke app
    # We get a single array result as chain_four waits for all publishers to finish
    # before operating on all elements published to topic_two as an array
    for _ in range(100):
        assert await app.ainvoke(2) == [13, 13]

    assert await asyncio.gather(*(app.ainvoke(2) for _ in range(100))) == [
        [13, 13] for _ in range(100)
    ]


async def test_invoke_join_then_call_other_pubsub(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    add_10_each = mocker.Mock(side_effect=lambda x: [y + 10 for y in x])

    inner_app = Pregel(
        Pregel.subscribe_to("input") | add_one | Pregel.send_to("output"),
        channels={
            "input": channels.LastValue(int),
            "output": channels.LastValue(int),
        },
        input="input",
        output="output",
    )

    chain_one = (
        Pregel.subscribe_to("input") | add_10_each | Pregel.send_to("inbox_one").map()
    )
    chain_two = (
        Pregel.subscribe_to("inbox_one")
        | inner_app.map()
        | sorted
        | Pregel.send_to("outbox_one")
    )
    chain_three = Pregel.subscribe_to("outbox_one") | sum | Pregel.send_to("output")

    app = Pregel(
        chain_one,
        chain_two,
        chain_three,
        channels={
            "input": channels.LastValue(int),
            "output": channels.LastValue(int),
            "inbox_one": channels.Inbox(int),
            "outbox_one": channels.LastValue(int),
        },
        input="input",
        output="output",
    )

    # Then invoke pubsub
    for _ in range(10):
        assert await app.ainvoke([2, 3]) == 27

    assert await asyncio.gather(*(app.ainvoke([2, 3]) for _ in range(10))) == [
        27 for _ in range(10)
    ]


async def test_invoke_two_processes_one_in_two_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    chain_one = (
        Pregel.subscribe_to("input")
        | add_one
        | Pregel.send_to(output=RunnablePassthrough(), between=RunnablePassthrough())
    )
    chain_two = Pregel.subscribe_to("between") | add_one | Pregel.send_to("output")

    app = Pregel(
        chain_one,
        chain_two,
        channels={
            "input": channels.LastValue(int),
            "output": channels.LastValue(int),
            "between": channels.LastValue(int),
        },
        input="input",
        output="output",
    )

    # Then invoke pubsub
    assert [c async for c in app.astream(2)] == [3, 4]


async def test_invoke_two_processes_no_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain_one = Pregel.subscribe_to("input") | add_one | Pregel.send_to("between")
    chain_two = Pregel.subscribe_to("between") | add_one

    app = Pregel(
        chain_one,
        chain_two,
        channels={
            "input": channels.LastValue(int),
            "output": channels.LastValue(int),
            "between": channels.LastValue(int),
        },
        input="input",
        output="output",
    )

    # Then invoke pubsub
    # It finishes executing (once no more messages being published)
    # but returns nothing, as nothing was published to OUT topic
    assert await app.ainvoke(2) is None
