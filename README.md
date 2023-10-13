# `permchain`

## Get started

`pip install permchain`

## Usage

```python
from permchain import Pregel, channels

value = channels.LastValue[str]("value")

grow_value = (
    Pregel.subscribe_to(value)
    | (lambda x: x + x)
    | Pregel.send_to({value: lambda x: x if len(x) < 10 else None})
)

pubsub = Pregel(grow_value, input=value, output=value)

assert pubsub.invoke("a") == "aaaaaaaa"
```

Check `examples` for more examples.

## Near-term Roadmap

- [ ] Iterate on API
  - [ ] do we want api to receive output from multiple channels in invoke()
  - [ ] do we want api to send input to multiple channels in invoke()
- [ ] Implement checkpointing
  - [ ] Save checkpoints at end of each step
  - [ ] Load checkpoint at start of invocation
  - [ ] API to specify storage backend and save key
- [ ] Add more examples
  - [ ] human in the loop
  - [ ] combine documents
  - [ ] agent executor
  - [ ] run over dataset
- [ ] Fault tolerance
  - [ ] Retry individual processes in a step
  - [ ] Retry entire step?
