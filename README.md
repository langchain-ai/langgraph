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
- [ ] Detect cycles (aka. infinite loops) and throw an error
  - [ ] Allow user to catch that error (by subcribing to an error topic?)
- [ ] Replace Queue data structure with a Log data structure (this will enable checking the status of the readers, etc.)
  - [ ] eg. https://anyio.readthedocs.io/en/3.x/streams.html
- [ ] Enable resuming PubSub from the "middle" of the computation
- [ ] Add "human in the loop" pattern
- [ ] Add "wait until topic X is done" pattern
- [ ] Add Redis-backed Connection implementation
