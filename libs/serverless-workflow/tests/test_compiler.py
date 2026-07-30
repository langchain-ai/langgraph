import pytest

from langgraph_serverless_workflow import ServerlessWorkflowCompiler


def test_inject_operation_end():
    dsl = """
id: greetflow
version: '1.0'
specVersion: '0.8'
start: InjectState
states:
  - name: InjectState
    type: inject
    data:
      who: world
    transition: GreetState
  - name: GreetState
    type: operation
    actions:
      - functionRef:
          refName: greet
    end: true
"""
    compiler = ServerlessWorkflowCompiler()
    compiler.register_function("greet", lambda **kw: {"msg": f"hi {kw['who']}"})
    graph = compiler.compile(dsl)

    result = graph.invoke({"data": {}})
    assert result["data"] == {"who": "world", "msg": "hi world"}


def test_str_function_ref_and_transition():
    dsl = """
id: t
version: '1.0'
specVersion: '0.8'
start: First
states:
  - name: First
    type: operation
    actions:
      - functionRef: double
    transition: Second
  - name: Second
    type: operation
    actions:
      - functionRef: double
    end: true
"""
    compiler = ServerlessWorkflowCompiler()
    compiler.register_function("double", lambda **kw: {"n": kw.get("n", 0) * 2})
    graph = compiler.compile(dsl)

    result = graph.invoke({"data": {"n": 1}})
    assert result["data"]["n"] == 4


def test_switch_branch_taken():
    dsl = """
id: s
version: '1.0'
specVersion: '0.8'
start: Check
states:
  - name: Check
    type: switch
    dataConditions:
      - condition: ".status == 'ok'"
        transition: Done
      - condition: ".status == 'err'"
        end: true
    defaultCondition:
        end: true
  - name: Done
    type: inject
    data:
      finished: true
    end: true
"""
    compiler = ServerlessWorkflowCompiler()
    graph = compiler.compile(dsl)

    ok = graph.invoke({"data": {"status": "ok"}})
    assert ok["data"]["finished"] is True
    assert "msg" not in ok["data"]

    err = graph.invoke({"data": {"status": "err"}})
    assert "finished" not in err["data"]


def test_switch_default_branch():
    dsl = """
id: sd
version: '1.0'
specVersion: '0.8'
start: Check
states:
  - name: Check
    type: switch
    dataConditions:
      - condition: ".x > 10"
        transition: High
    defaultCondition:
        transition: Low
  - name: High
    type: inject
    data:
      bucket: high
    end: true
  - name: Low
    type: inject
    data:
      bucket: low
    end: true
"""
    compiler = ServerlessWorkflowCompiler()
    graph = compiler.compile(dsl)

    high = graph.invoke({"data": {"x": 99}})
    assert high["data"]["bucket"] == "high"

    low = graph.invoke({"data": {"x": 1}})
    assert low["data"]["bucket"] == "low"


def test_implicit_start():
    dsl = """
id: is
version: '1.0'
specVersion: '0.8'
states:
  - name: Only
    type: inject
    data:
      v: 1
    end: true
"""
    compiler = ServerlessWorkflowCompiler()
    graph = compiler.compile(dsl)
    result = graph.invoke({"data": {}})
    assert result["data"]["v"] == 1


def test_sleep_state_uses_sleeper():
    dsl = """
id: sl
version: '1.0'
specVersion: '0.8'
start: Wait
states:
  - name: Wait
    type: sleep
    duration: PT2S
    transition: Done
  - name: Done
    type: inject
    data:
      ok: true
    end: true
"""
    slept = []
    compiler = ServerlessWorkflowCompiler()
    compiler.register_sleeper(lambda secs: slept.append(secs))
    graph = compiler.compile(dsl)

    result = graph.invoke({"data": {}})
    assert result["data"]["ok"] is True
    assert slept == [2.0]


def test_callback_state():
    dsl = """
id: cb
version: '1.0'
specVersion: '0.8'
start: Call
states:
  - name: Call
    type: callback
    action:
      functionRef:
        refName: prepare
    eventRef: callbackEvent
    transition: Done
  - name: Done
    type: inject
    data:
      finished: true
    end: true
"""
    compiler = ServerlessWorkflowCompiler()
    compiler.register_function("prepare", lambda **kw: {"req": "hello"})
    compiler.register_callback_handler("callbackEvent", lambda current: {"resp": f"ack:{current['req']}"})
    graph = compiler.compile(dsl)

    result = graph.invoke({"data": {}})
    assert result["data"]["req"] == "hello"
    assert result["data"]["resp"] == "ack:hello"
    assert result["data"]["finished"] is True


def test_event_state():
    dsl = """
id: ev
version: '1.0'
specVersion: '0.8'
start: Listen
states:
  - name: Listen
    type: event
    onEvents:
      - eventRefs:
          - triggerEvent
        actions:
          - functionRef:
              refName: echo
    transition: Done
  - name: Done
    type: inject
    data:
      done: true
    end: true
"""
    compiler = ServerlessWorkflowCompiler()
    compiler.register_event_handler("triggerEvent", lambda current: {"payload": 42})
    compiler.register_function("echo", lambda **kw: {"echoed": kw.get("payload")})
    graph = compiler.compile(dsl)

    result = graph.invoke({"data": {}})
    assert result["data"]["payload"] == 42
    assert result["data"]["echoed"] == 42
    assert result["data"]["done"] is True


def test_foreach_state_sequential():
    dsl = """
id: fe
version: '1.0'
specVersion: '0.8'
start: Map
states:
  - name: Map
    type: foreach
    mode: sequential
    inputCollection: ".items"
    iterationParam: item
    outputCollection: ".results"
    actions:
      - functionRef:
          refName: square
    end: true
"""
    compiler = ServerlessWorkflowCompiler()
    # `square` reads the iteration param from the forwarded state; actions
    # without explicit arguments receive the whole state as kwargs, so accept
    # `**kw` and pull out the iteration value.
    compiler.register_function("square", lambda **kw: {"value": kw["item"] ** 2})
    graph = compiler.compile(dsl)

    result = graph.invoke({"data": {"items": [1, 2, 3]}})
    assert result["data"]["results"] == [
        {"value": 1},
        {"value": 4},
        {"value": 9},
    ]


def test_parallel_state():
    dsl = """
id: pl
version: '1.0'
specVersion: '0.8'
start: Run
states:
  - name: Run
    type: parallel
    branches:
      - name: b1
        actions:
          - functionRef:
              refName: produceA
      - name: b2
        actions:
          - functionRef:
              refName: produceB
    transition: Done
  - name: Done
    type: inject
    data:
      done: true
    end: true
"""
    compiler = ServerlessWorkflowCompiler()
    compiler.register_function("produceA", lambda **kw: {"a": 1})
    compiler.register_function("produceB", lambda **kw: {"b": 2})
    graph = compiler.compile(dsl)

    result = graph.invoke({"data": {}})
    assert result["data"]["a"] == 1
    assert result["data"]["b"] == 2
    assert result["data"]["done"] is True


def test_foreach_state_parallel():
    dsl = """
id: fep
version: '1.0'
specVersion: '0.8'
start: Map
states:
  - name: Map
    type: foreach
    mode: parallel
    inputCollection: ".items"
    iterationParam: item
    outputCollection: ".results"
    actions:
      - functionRef:
          refName: square
    end: true
"""
    compiler = ServerlessWorkflowCompiler()
    compiler.register_function("square", lambda **kw: {"value": kw["item"] ** 2})
    graph = compiler.compile(dsl)

    result = graph.invoke({"data": {"items": [1, 2, 3]}})
    assert sorted(r["value"] for r in result["data"]["results"]) == [1, 4, 9]


def test_action_condition_skips_action():
    dsl = """
id: cond
version: '1.0'
specVersion: '0.8'
start: Op
states:
  - name: Op
    type: operation
    actions:
      - functionRef:
          refName: never
        condition: ".run == false"
      - functionRef:
          refName: echo
        condition: ".run == true"
    end: true
"""
    compiler = ServerlessWorkflowCompiler()
    compiler.register_function("never", lambda **kw: {"x": 1})
    compiler.register_function("echo", lambda **kw: {"ran": True})
    graph = compiler.compile(dsl)

    result = graph.invoke({"data": {"run": True}})
    assert result["data"].get("ran") is True
    assert "x" not in result["data"]


def test_action_sleep_before_after():
    dsl = """
id: aslp
version: '1.0'
specVersion: '0.8'
start: Op
states:
  - name: Op
    type: operation
    actions:
      - functionRef:
          refName: tick
        sleep:
          before: PT1S
          after: PT2S
    end: true
"""
    slept = []
    compiler = ServerlessWorkflowCompiler()
    compiler.register_sleeper(lambda secs: slept.append(secs))
    compiler.register_function("tick", lambda **kw: {"v": 1})
    graph = compiler.compile(dsl)

    result = graph.invoke({"data": {}})
    assert result["data"]["v"] == 1
    assert slept == [1.0, 2.0]


def test_unsupported_state_type_raises():
    # The serverlessworkflow SDK validates state types at parse time, so an
    # unknown type can only reach the compiler via a hydrated state object.
    # Exercise the guard directly with a lightweight stub.
    class _StubState:
        type = "bogus"
        name = "Weird"

    compiler = ServerlessWorkflowCompiler()
    with pytest.raises(ValueError, match="Unsupported state type"):
        compiler._build_node_function(_StubState())


def test_missing_function_raises():
    dsl = """
id: mf
version: '1.0'
specVersion: '0.8'
start: Op
states:
  - name: Op
    type: operation
    actions:
      - functionRef:
          refName: nope
    end: true
"""
    compiler = ServerlessWorkflowCompiler()
    graph = compiler.compile(dsl)
    with pytest.raises(KeyError, match="nope"):
        graph.invoke({"data": {}})


def test_non_identifier_state_key_does_not_crash():
    dsl = """
id: nii
version: '1.0'
specVersion: '0.8'
start: Op
states:
  - name: Op
    type: operation
    actions:
      - functionRef:
          refName: read
    end: true
"""
    compiler = ServerlessWorkflowCompiler()
    compiler.register_function("read", lambda **kw: {"got": kw.get("good")})
    graph = compiler.compile(dsl)

    # "bad-key" and "123" are not valid identifiers and must be dropped
    # rather than crashing func(**kwargs); "good" is forwarded.
    result = graph.invoke({"data": {"bad-key": 1, "123": 2, "good": 3}})
    assert result["data"]["got"] == 3
