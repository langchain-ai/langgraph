import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Generator

import pytest
from langchain.schema.runnable import RunnablePassthrough
from pytest_mock import MockerFixture

from permchain import Pregel, channels
from permchain.pregel import PregelInvoke


def test_invoke_single_process_in_out(mocker: MockerFixture) -> None:
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

    assert app.input_schema.schema() == {"title": "PregelInput", "type": "integer"}
    assert app.output_schema.schema() == {"title": "PregelOutput", "type": "integer"}
    assert app.invoke(2) == 3


def test_invoke_single_process_in_out_dict(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = Pregel.subscribe_to("input") | add_one | Pregel.send_to("output")

    app = Pregel(
        (chain,),
        channels={
            "input": channels.LastValue(int),
            "output": channels.LastValue(int),
        },
        input="input",
        output=["output"],
    )

    assert app.input_schema.schema() == {"title": "PregelInput", "type": "integer"}
    assert app.output_schema.schema() == {
        "title": "PregelOutput",
        "type": "object",
        "properties": {"output": {"title": "Output", "type": "integer"}},
    }
    assert app.invoke(2) == {"output": 3}


def test_invoke_single_process_in_dict_out_dict(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = Pregel.subscribe_to("input") | add_one | Pregel.send_to("output")

    app = Pregel(
        (chain,),
        channels={
            "input": channels.LastValue(int),
            "output": channels.LastValue(int),
        },
        input=["input"],
        output=["output"],
    )

    assert app.input_schema.schema() == {
        "title": "PregelInput",
        "type": "object",
        "properties": {"input": {"title": "Input", "type": "integer"}},
    }
    assert app.output_schema.schema() == {
        "title": "PregelOutput",
        "type": "object",
        "properties": {"output": {"title": "Output", "type": "integer"}},
    }
    assert app.invoke({"input": 2}) == {"output": 3}


def test_invoke_two_processes_in_out(mocker: MockerFixture) -> None:
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

    assert app.invoke(2) == 4


def test_invoke_two_processes_in_dict_out(mocker: MockerFixture) -> None:
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
        input=["input", "inbox"],
        output="output",
    )

    assert [*app.stream({"input": 2, "inbox": 12})] == [13, 4]  # [12 + 1, 2 + 1 + 1]


def test_batch_two_processes_in_out() -> None:
    def add_one_with_delay(inp: int) -> int:
        time.sleep(inp / 10)
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

    assert app.batch([3, 2, 1, 3, 5]) == [5, 4, 3, 5, 7]


def test_invoke_many_processes_in_out(mocker: MockerFixture) -> None:
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

    for _ in range(10):
        assert app.invoke(2, {"recursion_limit": test_size}) == 2 + test_size

    with ThreadPoolExecutor() as executor:
        assert [
            *executor.map(app.invoke, [2] * 10, [{"recursion_limit": test_size}] * 10)
        ] == [2 + test_size] * 10


def test_batch_many_processes_in_out(mocker: MockerFixture) -> None:
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

    for _ in range(10):
        assert app.batch([2, 1, 3, 4, 5], {"recursion_limit": test_size}) == [
            2 + test_size,
            1 + test_size,
            3 + test_size,
            4 + test_size,
            5 + test_size,
        ]

    with ThreadPoolExecutor() as executor:
        assert [
            *executor.map(
                app.batch, [[2, 1, 3, 4, 5]] * 10, [{"recursion_limit": test_size}] * 10
            )
        ] == [
            [2 + test_size, 1 + test_size, 3 + test_size, 4 + test_size, 5 + test_size]
        ] * 10


def test_invoke_two_processes_two_in_two_out_invalid(mocker: MockerFixture) -> None:
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
        app.invoke(2)


def test_invoke_two_processes_two_in_two_out_valid(mocker: MockerFixture) -> None:
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
    assert app.invoke(2) == (3, 3)


def test_invoke_two_processes_two_in_join_two_out(mocker: MockerFixture) -> None:
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
        assert app.invoke(2) == [13, 13]

    with ThreadPoolExecutor() as executor:
        assert [*executor.map(app.invoke, [2] * 100)] == [[13, 13]] * 100


def test_invoke_join_then_call_other_app(mocker: MockerFixture) -> None:
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

    for _ in range(10):
        assert app.invoke([2, 3]) == 27

    with ThreadPoolExecutor() as executor:
        assert [*executor.map(app.invoke, [[2, 3]] * 10)] == [27] * 10


def test_invoke_two_processes_one_in_two_out(mocker: MockerFixture) -> None:
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

    assert [c for c in app.stream(2)] == [3, 4]


def test_invoke_two_processes_no_out(mocker: MockerFixture) -> None:
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

    # It finishes executing (once no more messages being published)
    # but returns nothing, as nothing was published to OUT topic
    assert app.invoke(2) is None


def test_invoke_two_processes_no_in(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    chain_one = Pregel.subscribe_to("between") | add_one | Pregel.send_to("output")
    chain_two = Pregel.subscribe_to("between") | add_one

    with pytest.raises(ValueError):
        Pregel(
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


def test_channel_enter_exit_timing(mocker: MockerFixture) -> None:
    setup = mocker.Mock()
    cleanup = mocker.Mock()

    @contextmanager
    def an_int() -> Generator[int, None, None]:
        setup()
        try:
            yield 5
        finally:
            cleanup()

    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain_one = Pregel.subscribe_to("input") | add_one | Pregel.send_to("inbox")
    chain_two = Pregel.subscribe_to_each("inbox") | add_one | Pregel.send_to("output")

    app = Pregel(
        [chain_one, chain_two],
        channels={
            "input": channels.LastValue(int),
            "output": channels.LastValue(int),
            "inbox": channels.Inbox(int),
            "ctx": channels.ContextManager(an_int, typ=int),
        },
        input="input",
        output=["inbox", "output"],
    )

    assert setup.call_count == 0
    assert cleanup.call_count == 0
    for i, chunk in enumerate(app.stream(2)):
        assert setup.call_count == 1, "Expected setup to be called once"
        assert cleanup.call_count == 0, "Expected cleanup to not be called yet"
        if i == 0:
            assert chunk == {"inbox": (3,)}
        elif i == 1:
            assert chunk == {"output": 4}
        else:
            assert False, "Expected only two chunks"
    assert cleanup.call_count == 1, "Expected cleanup to be called once"
