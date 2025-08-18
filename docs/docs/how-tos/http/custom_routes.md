# How to add custom routes

When deploying agents to LangGraph platform, your server automatically exposes routes for creating runs and threads, interacting with the long-term memory store, managing configurable assistants, and other core functionality ([see all default API endpoints](../../cloud/reference/api/api_ref.md)).

You can add custom routes by providing your own [`Starlette`](https://www.starlette.io/applications/) app (including [`FastAPI`](https://fastapi.tiangolo.com/), [`FastHTML`](https://fastht.ml/) and other compatible apps). You make LangGraph Platform aware of this by providing a path to the app in your `langgraph.json` configuration file.

Defining a custom app object lets you add any routes you'd like, so you can do anything from adding a `/login` endpoint to writing an entire full-stack web-app, all deployed in a single LangGraph Server.

Below is an example using FastAPI.

## Create app

Starting from an **existing** LangGraph Platform application, add the following custom route code to your webapp file. If you are starting from scratch, you can create a new app from a template using the CLI.

```bash
langgraph new --template=new-langgraph-project-python my_new_project
```

Once you have a LangGraph project, add the following app code:

```python
# ./src/agent/webapp.py
from fastapi import FastAPI

# highlight-next-line
app = FastAPI()


@app.get("/hello")
def read_root():
    return {"Hello": "World"}

```

## Configure `langgraph.json`

Add the following to your `langgraph.json` configuration file. Make sure the path points to the FastAPI application instance `app` in the `webapp.py` file you created above.

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/agent/graph.py:graph"
  },
  "env": ".env",
  "http": {
    "app": "./src/agent/webapp.py:app"
  }
  // Other configuration options like auth, store, etc.
}
```

## Start server

Test the server out locally:

```bash
langgraph dev --no-browser
```

If you navigate to `localhost:2024/hello` in your browser (`2024` is the default development port), you should see the `/hello` endpoint returning `{"Hello": "World"}`.

!!! note "Shadowing default endpoints"

    The routes you create in the app are given priority over the system defaults, meaning you can shadow and redefine the behavior of any default endpoint.

## Deploying

You can deploy this app as-is to LangGraph Platform or to your self-hosted platform.

## Next steps

Now that you've added a custom route to your deployment, you can use this same technique to further customize how your server behaves, such as defining custom [custom middleware](./custom_middleware.md) and [custom lifespan events](./custom_lifespan.md).
