# Quick Start
This quick start guide will cover how to develop an application for LangGraph Cloud, run it locally in Docker, and call the APIs to invoke a graph.


## Set up local files

1. Create a new application with the following directory and files:

        <my-app>/
        |-- agent.py            # code for your LangGraph agent
        |-- requirements.txt    # Python packages required for your graph
        |-- langgraph.json      # configuration file for LangGraph
        |-- .env                # environment files with API keys

2. The `agent.py` file should contain Python code for defining your graph. The following code is a simple example, the important thing is that at somepoint in your file you compile your graph and assign that Runnable to a variable (in this case the `graph` variable). 

    ```python
    from langchain_anthropic import ChatAnthropic
    from langgraph.graph import END, MessageGraph

    model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

    graph_workflow = MessageGraph()

    graph_workflow.add_node("agent", model)
    graph_workflow.add_edge("agent", END)
    graph_workflow.set_entry_point("agent")

    graph = graph_workflow.compile()
    ```

3. The `requirements.txt` file should contain any dependencies for your graph(s). In this case we only require two packages for our graph to run:

        langgraph
        langchain_anthropic

4. The `langgraph.json` file is a configuration file that describes what graph you are going to host. It is important to note that you can host multiple graphs at a time. In this case we only host one: the compiled `graph` object from `agent.py`, but we could have defined multiple different graphs within `agent.py` to host, or written multiple python files to each host one or more graphs. Each graph you wish to host should have a unique identifier.

    ```json
    {
        "dependencies": ["."],
        "graphs": {
            "agent": "./agent.py:graph"
        },
        "env": ".env"
    }
    ```

    Learn more about the LangGraph CLI configuration file [here](./reference/cli.md#configuration-file).

5. The `.env` file should contain the environment variables that are needed to run your graph. In this case we just need to specify the OpenAI API key, as well as the authentication type for langgraph.  

        ANTHROPIC_API_KEY=<add your key here>
        LANGGRAPH_AUTH_TYPE=noop

    !!! warning "Disable Authentication"
        When testing locally, set `LANGGRAPH_AUTH_TYPE` to `noop` to disable authentication.

Now that we have set everything up on our local file system, we are ready to host our graph. 

## Run Locally

1. Install the LangGraph CLI by using the following steps:
    1. Ensure that Docker is installed (confirm by running `docker --version` in your terminal).
    2. Install the `langgraph-cli` Python package (e.g. `pip install langgraph-cli`).
    3. Run the command `langgraph --help` to confirm that the CLI is installed.

2. Run the following command to start the API server in Docker:

        langgraph up -c langgraph.json

3. The API server is now running at `http://localhost:8123`. Navigate to [`http://localhost:8123/docs`](http://localhost:8123/docs) to view the API docs.

4. You can now test that your deployment is working as intended by invoking some of the cURL commands from the API docs.

First, let's test that the assistant we are hosting is indeed retrievable by the API, which we can do by using the "assistants/search" endpoint:

    curl --request POST \
        --url http://localhost:8123/assistants/search \
        --header 'Content-Type: application/json' \
        --data '{
        "metadata": {},
        "limit": 10,
        "offset": 0
    }

If the hosting is working as expected, you should receive a 200 response which looks something like this example response:

    [
        {
            "assistant_id": "123e4567-e89b-12d3-a456-426614174000",
            "graph_id": "agent",
            "config": {
            "tags": [
                "…"
            ],
            "recursion_limit": 1,
            "configurable": {}
            },
            "created_at": "2024-06-24T19:21:47.514Z",
            "updated_at": "2024-06-24T19:21:47.514Z",
            "metadata": {}
        }
    ]

Once you have verified that this step is working, you can test out that invoking your hosted graphs works as intended without any bugs. You can do this by calling the a version of the following cURL command:

    curl --request POST \
        --url http://localhost:8123/runs/stream \
        --header 'Content-Type: application/json' \
        --data '{
        "assistant_id": "123e4567-e89b-12d3-a456-426614174000",
        "input": {     
            "messages": [
            {               
                "role": "user",
                "content": "How are you?"
            }           
            ]       
        },
        "metadata": {},
        "config": {
            "configurable": {}
        },
        "multitask_strategy": "reject",
        "stream_mode": [
            "values"
        ]
    }'

Make sure to edit the `input` and `assistant_id` fields to match what assistant you want to test. If you receive a 200 response then congratulations your graph has run successfully and you are ready to move on to hosting on LangGraph Cloud!

## Deploy to Cloud

### Push your code to GitHub

Create a git repo in the `<my-app>` directory, and verify it’s existence. You can use the GitHub CLI if you like, or just create a repo manually.

### Deploy from GitHub with LangGraph Cloud

Once you have created your github repository with a Python file containing your compiled graph as well as a `langgraph.json` file containing the configuration for hosting your graph, you can head over to LangSmith and click on the 🚀 icon on the left navbar to create a new deployment. Then click the `+ New Deployment` button. 

![Langsmith Workflow](./img/cloud_deployment.png)

***If you have not deployed to LangGraph Cloud before:*** there will be a button that shows up saying Import from GitHub. You’ll need to follow that flow to connect LangGraph Cloud to GitHub.

***Once you have set up your GitHub connection:*** the new deployment page will look as follows

![Screenshot 2024-06-11 at 1.17.03 PM.png](./deployment/img/deployment_page.png)

To deploy your application, you should do the following:

1. Select your GitHub username or organization from the selector
2. Search for your repo to deploy in the search bar and select it
3. Choose any name
4. In the `LangGraph API config file` field, enter the path to your `langgraph.json` file (if left blank langsmith will automatically search for it on deployment)
5. For Git Reference, you can select either the git branch for the code you want to deploy, or the exact commit SHA. 
6. If your chain relies on environment variables (for example, an OPENAI_API_KEY), add those in. They will be propagated to the underlying server so your code can access them.

Putting this all together, you should have something as follows for your deployment details:

![Screenshot 2024-06-11 at 1.21.52 PM.png](./deployment/img/deploy_filled_out.png)

Hit `Submit` and your application will start deploying!

## Inspect Traces + Monitor Service

### Deployments View

After your deployment is complete, your deployments page should look as follows:

![Screenshot 2024-06-11 at 2.03.34 PM.png](./deployment/img/deployed_page.png)

You can see that by default, you get access to the `Trace Count` monitoring chart and `Recent Traces` run view. These are powered by LangSmith. 

You can click on `All Charts` to view all monitoring info for your server, or click on `See tracing project` to get more information on an individual trace.

### Access the Docs

You can access the docs by clicking on the API docs link, which should send you to a page that looks like this:

![Screenshot 2024-06-19 at 2.27.24 PM.png](./deployment/img/api_page.png)

You won’t actually be able to test any of the API endpoints without authorizing first. To do so, click on the Authorize button in the top right corner, input your `LANGCHAIN_API_KEY`  in the `API Key` box, and then click `Authorize`  to finish the process. You should now be able to select any of the API endpoints, click `Try it out` , enter the parameters you would like to pass, and then click `Execute` to view the results of the API call.

## Interact with your deployment via LangGraph Studio

If you click on your deployment you should see a blue button in the top right that says `LangGraph Studio`. Clicking on this button will take you to a page that looks like this:

![Screenshot 2024-06-11 at 2.51.51 PM.png](./deployment/img/graph_visualization.png)

On this page you can test out your graph by passing in starting states and clicking `Start Run` (this should behave identically to calling `.invoke`). You will then be able to look into the execution thread for each run and explore the steps your graph is taking to produce its output.

## Use with the SDK

Once you have tested that your hosted graph works as expected using LangGraph Studio, you can start using your hosted graph all over your organization by using the LangGraph SDK. Let's see how we can access our hosted graph and execute our run from a python file. 

First, make sure you have the SDK installed by calling `pip install langgraph_sdk`.

The first thing to do when using the SDK is to setup our client, access our assistant, and create a thread to execute a run on:

```python
from langgraph_sdk import get_client

# get top-level LangGraphClient
# this is the url for self-hosted LangGraph, replace with your hosted URL if applicable
client = get_client(url="http://localhost:8123")

# Search all hosted graphs
assistants = await client.assistants.search()
# In this example we select the first assistant since we are only hosting a single graph
assistant = assistants[0]

# We create a thread for tracking the state of our run
thread = await client.threads.create()
```

We can then execute a run on the thread:

```python
input = {"messages":[{"role": "user", "content": "Hello! My name is Bagatur and I am 26 years old."}]}

async for chunk in client.runs.stream(
        thread['thread_id'],
        assistant["assistant_id"],
        input=input,
        stream_mode="updates",
    ):
    if chunk.data and "run_id" not in chunk.data:
            print(chunk.data)
```

    {'agent': {'messages': [{'content': "Hi Bagatur! It's nice to meet you. How can I assist you today?", 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_9cb5d38cf7'}, 'type': 'ai', 'name': None, 'id': 'run-c89118b7-1b1e-42b9-a85d-c43fe99881cd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}


You can learn more about the Python SDK in [this how-to guide](./sdk/python_sdk.ipynb), and read up on the Javascript SDK in [this how-to guide](./sdk/js_sdk.ipynb)

## What's Next

Congratulations! If you've worked your way through this tutorial you are well on your way to becoming a LangGraph Cloud expert. Here are some other resources to check out to help you out on the path to expertise:

### LangGraph Cloud How-tos

If you want to learn more about streaming from hosted graphs, check out the Streaming [how-to guides](https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/stream_values/).

To learn more about double-texting and all the ways you can handle it in your application, read up on these [how-to guides](https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/interrupt_concurrent/).

To learn about how to include different human-in-the-loop behavior in your graph, take a look at [these how-tos](https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/human_in_the_loop_breakpoint/).

### LangGraph Tutorials

Before hosting, you have to write a graph to host. Here are some tutorials to get you more comfortable with writing LangGraph graphs and give you inspiration for the types of graphs you want to host.

[This tutorial](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/) walks you through how to write a customer support bot using LangGraph.

If you are interested in writing a SQL agent, check out [this tutorial](https://langchain-ai.github.io/langgraph/tutorials/sql-agent/).

Check out the [LangGraph tutorials](https://langchain-ai.github.io/langgraph/tutorials/) page to read about more exciting use cases.

