# `permchain`

## Get started

`pip install permchain`

## Usage

```python
from permchain import InMemoryPubSubConnection, PubSub, Topic

topic_one = Topic("one")
chain_one = Topic.IN.subscribe() | (lambda x: x + 'b') | topic_one.publish()
chain_two = topic_one.subscribe() | (lambda x: x + 'c') | Topic.OUT.publish()

conn = InMemoryPubSubConnection()
pubsub = PubSub(processes=(chain_one, chain_two), connection=conn)

assert pubsub.invoke('a') == ['abc']
```

Check `tests` and `examples` for more examples.

## Near-term Roadmap

- [x] Add initial retry support (pending changes in `langchain`)
- [x] Implement OUT as regular topic
- [x] Implement IN as regular topic
- [x] Add Connection.peek() to monitor past messages from all topics
- [x] Enable resuming PubSub from the "middle" of the computation
- [x] Add test for .peek()
- [ ] Move tracking of inflight processes/messages to Connection
  - [ ] Use this to build retry mechanism, where any inflight messages are moved back to the respective topics when restarting
- [ ] Detect cycles (aka. infinite loops) and throw an error
  - [ ] Allow user to catch that error (by subcribing to an error topic?)
- [ ] Add "human in the loop" pattern, one of the two below
  - [ ] Example with one permchain, which runs until it produces either 1. request for input or 2. output. The consumer code then gets the needed info, and restarts the permchain with answer, and same state id
  - [ ] Allow interrupting execution by breaking out of the iterator returned by .stream()
    - [ ] Build example showing a simple "human in the loop" pattern using this, ie. if a certain message asking for input is published the consumer of the iterator breaks out, does something and then restarts it
- [ ] Add "wait until topic X is done" pattern, aka. `Topic.reduce()`
- [ ] Add Redis-backed Connection implementation
