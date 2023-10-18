# `permchain`

## Get started

`pip install permchain`

## Usage

```python
from permchain import Pregel, channels

grow_value = (
    Pregel.subscribe_to("value")
    | (lambda x: x + x)
    | Pregel.send_to(value=lambda x: x if len(x) < 10 else None)
)

app = Pregel(
    grow_value,
    channels={"value": channels.LastValue[str]()},
    input="value",
    output="value",
)

assert app.invoke("a") == "aaaaaaaa"

```

Check `examples` for more examples.

## Near-term Roadmap

- [x] Iterate on API
  - [x] do we want api to receive output from multiple channels in invoke()
  - [x] do we want api to send input to multiple channels in invoke()
  - [x] Finish updating tests to new API
- [ ] Implement input_schema and output_schema in Pregel
- [ ] More tests
  - [ ] Test different input and output types (str, str sequence, None)
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
