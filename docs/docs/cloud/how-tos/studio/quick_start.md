!!! info "Prerequisites"

    - [LangGraph Studio Overview](../../../concepts/langgraph_studio.md)

LangGraph Studio supports connecting to two types of graphs:

- Graphs deployed on [LangGraph Platform](../../../cloud/quick_start.md)
- Graphs running locally via the [LangGraph Server](../../../tutorials/langgraph-platform/local-server.md).

## Deployed Application

For applications that are deployed on LangGraph Platform, you can access Studio as part of that deployment. To do so, navigate to the deployment in LangGraph Platform within the LangSmith UI and click the "LangGraph Studio" button.

This will load the Studio UI connected to your live deployment, allowing you to create, read, and update the [threads](../../concepts/threads.md), [assistants](../../../concepts/assistants.md), and [memory](../../../concepts//memory.md) in that deployment.

## Local Development Server

To test your locally running application using LangGraph Studio, ensure your application is set up following [this guide](https://langchain-ai.github.io/langgraph/cloud/deployment/setup/).

Next, install the [LangGraph CLI](../../../concepts/langgraph_cli.md):

```
pip install -U "langgraph-cli[inmem]"
```

and run:

```
langgraph dev
```

!!! warning "Browser Compatibility"
    Safari blocks `localhost` connections to Studio. To work around this, run the above command with `--tunnel` to access Studio via a secure tunnel.

This will start the LangGraph Server locally, running in-memory. The server will run in watch mode, listening for and automatically restarting on code changes. Read this [reference](https://langchain-ai.github.io/langgraph/cloud/reference/cli/#dev) to learn about all the options for starting the API server.

If successful, you will see the following logs:

> Ready!
>
> - API: [http://localhost:2024](http://localhost:2024/)
>
> - Docs: http://localhost:2024/docs
>
> - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

Once running, you will automatically be directed to LangGraph Studio. 


For an already running server, access Studio by either:

 1. Directly navigate to the following URL: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`. 
 2. Within LangSmith, navigate to the LangGraph Platform Deployments tab, click the "LangGraph Studio" button, enter `http://127.0.0.1:2024` and click "Connect".
 
 If running your server at a different host or port, simply update the `baseUrl` to match.

### (Optional) Attach a debugger

For step-by-step debugging with breakpoints and variable inspection:

```bash
# Install debugpy package
pip install debugpy

# Start server with debugging enabled
langgraph dev --debug-port 5678
```

Then attach your preferred debugger:

=== "VS Code"
    Add this configuration to `launch.json`:
    ```json
      {
        "name": "Attach to LangGraph",
        "type": "debugpy",
        "request": "attach",
        "connect": {
          "host": "0.0.0.0",
          "port": 5678
        }
      }
    ```
    Specify the port number you chose in the previous step.

=== "PyCharm" 
    1. Go to Run → Edit Configurations 
    2. Click + and select "Python Debug Server" 
    3. Set IDE host name: `localhost` 
    4. Set port: `5678` (or the port number you chose in the previous step) 
    5. Click "OK" and start debugging


## Next steps

See the following how-tos for more information on how to use Studio:

- [How to manage Assistants](../invoke_studio.md)
- [How to manage Threads](../threads_studio.md)
- [How to create datasets](../datasets_studio.md)
- [How to prompt engineer](../iterate_graph_studio.md)
- [How to locally debug remote traces](../clone_traces_studio.md)
