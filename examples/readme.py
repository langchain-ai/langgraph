from permchain import Channel, Pregel
from permchain.channels import LastValue

grow_value = (
    Channel.subscribe_to("value")
    | (lambda x: x + x)
    | Channel.write_to(value=lambda x: x if len(x) < 10 else None)
)

app = Pregel(
    chains={"grow_value": grow_value},
    channels={"value": LastValue(str)},
    input="value",
    output="value",
)

assert app.invoke("a") == "aaaaaaaa"
