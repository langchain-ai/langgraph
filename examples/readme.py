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
