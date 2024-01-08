from langgraph.pregel import Channel, Pregel

grow_value = (
    Channel.subscribe_to("value")
    | (lambda x: x + x)
    | Channel.write_to(value=lambda x: x if len(x) < 10 else None)
)

app = Pregel(
    chains={"grow_value": grow_value},
    input="value",
    output="value",
)

assert app.invoke("a") == "aaaaaaaa"
