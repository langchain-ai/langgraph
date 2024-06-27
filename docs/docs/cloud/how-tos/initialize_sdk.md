# LangGraph SDK

Once your graph has been deployed using LangGraph cloud, interacting with it is made simple by using the LangGraph SDK.

Here are the steps to getting started with the SDK:

1. Download the SDK using either `pip install langgraph_sdk` for python, or `npm install @langchain/langgraph-sdk`
2. Ensure that the `LANGCHAIN_API_KEY` environment variable is set. To confirm that it is, you can call `echo $LANGCHAIN_API_KEY` from the terminal.
3. Start the client with your URL. To find your URL, go to your deployment in LangGraph Cloud and find the url that links to the docs. You want to copy that entire url except for the "/docs" part, as shown in this screenshot: ![URL](./img/sdk_url.png)

4. Then initialize the client as follows: 

Python
```
from langgraph_sdk import get_client

client = get_client(url="your-deployment-url")
``` 

Javacsript

```
import { Client } from "@langchain/langgraph-sdk";

const client = new Client({apiUrl:"whatever-your-url-is"});
```

5. Read the [SDK reference](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/) to learn about all the ways you can use it.