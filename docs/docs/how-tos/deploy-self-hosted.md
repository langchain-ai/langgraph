# How to do a Self-hosted deployment of LangGraph

!!! info "Prerequisites"

    - [Application Structure](../concepts/application_structure.md)
    - [Deployment Options](../concepts/deployment_options.md)

This how-to guide will walk you through how to create a docker image from an existing LangGraph application, so you can deploy it on your own infrastructure.

## How it works

With the self-hosted deployment option, you are responsible for managing the infrastructure, including setting up and maintaining necessary databases, Redis instances, and other services.

You will need to do the following:

1. Deploy Redis and Postgres instances on your own infrastructure.
2. Build a docker image with the [LangGraph Server](../concepts/langgraph_server.md) using the [LangGraph CLI](../concepts/langgraph_cli.md).
3. Deploy a web server that will run the docker image and pass in the necessary environment variables.

## Helm Chart

If you would like to deploy LangGraph Cloud on Kubernetes, you can use this [Helm chart](https://github.com/langchain-ai/helm/blob/main/charts/langgraph-cloud/README.md).

## Environment Variables

You will eventually need to pass in the following environment variables to the LangGraph Deploy server:

- `REDIS_URI`: Connection details to a Redis instance. Redis will be used as a pub-sub broker to enable streaming real time output from background runs. The value of `REDIS_URI` must be a valid [Redis connection URI](https://redis-py.readthedocs.io/en/stable/connections.html#redis.Redis.from_url).

    !!! Note "Shared Redis Instance"
        Multiple self-hosted deployments can share the same Redis instance. For example, for `Deployment A`, `REDIS_URI` can be set to `redis://<hostname_1>:<port>/1` and for `Deployment B`, `REDIS_URI` can be set to `redis://<hostname_1>:<port>/2`.

        `1` and `2` are different database numbers within the same instance, but `<hostname_1>` is shared. **The same database number cannot be used for separate deployments**.

- `DATABASE_URI`: Postgres connection details. Postgres will be used to store assistants, threads, runs, persist thread state and long term memory, and to manage the state of the background task queue with 'exactly once' semantics. The value of `DATABASE_URI` must be a valid [Postgres connection URI](https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING-URIS).

    !!! Note "Shared Postgres Instance"
        Multiple self-hosted deployments can share the same Postgres instance. For example, for `Deployment A`, `DATABASE_URI` can be set to `postgres://<user>:<password>@/<database_name_1>?host=<hostname_1>` and for `Deployment B`, `DATABASE_URI` can be set to `postgres://<user>:<password>@/<database_name_2>?host=<hostname_1>`.

        `<database_name_1>` and `database_name_2` are different databases within the same instance, but `<hostname_1>` is shared. **The same database cannot be used for separate deployments**.

- `LANGSMITH_API_KEY`: (If using [Self-Hosted Lite](../concepts/deployment_options.md#self-hosted-lite)) LangSmith API key. This will be used to authenticate ONCE at server start up.
- `LANGGRAPH_CLOUD_LICENSE_KEY`: (If using [Self-Hosted Enterprise](../concepts/deployment_options.md#self-hosted-enterprise)) LangGraph Platform license key. This will be used to authenticate ONCE at server start up.
- `LANGCHAIN_ENDPOINT`: To send traces to a [self-hosted LangSmith](https://docs.smith.langchain.com/self_hosting) instance, set `LANGCHAIN_ENDPOINT` to the hostname of the self-hosted LangSmith instance.


## Build the Docker Image

Please read the [Application Structure](../concepts/application_structure.md) guide to understand how to structure your LangGraph application.

If the application is structured correctly, you can build a docker image with the LangGraph Deploy server.

To build the docker image, you first need to install the CLI:

```shell
pip install -U langgraph-cli
```

You can then use:

```
langgraph build -t my-image
```

This will build a docker image with the LangGraph Deploy server. The `-t my-image` is used to tag the image with a name.

When running this server, you need to pass three environment variables:

## Running the application locally

### Using Docker

```shell
docker run \
    --env-file .env \
    -p 8123:8000 \
    -e REDIS_URI="foo" \
    -e DATABASE_URI="bar" \
    -e LANGSMITH_API_KEY="baz" \
    my-image
```

If you want to run this quickly without setting up a separate Redis and Postgres instance, you can use this docker compose file.

!!! note

    * You need to replace `my-image` with the name of the image you built in the previous step (from `langgraph build`).
    and you should provide appropriate values for `REDIS_URI`, `DATABASE_URI`, and `LANGSMITH_API_KEY`.
    * If your application requires additional environment variables, you can pass them in a similar way.
    * If using [Self-Hosted Enterprise](../concepts/deployment_options.md#self-hosted-enterprise), you must provide `LANGGRAPH_CLOUD_LICENSE_KEY` as an additional environment variable.


### Using Docker Compose

```yml
volumes:
    langgraph-data:
        driver: local
services:
    langgraph-redis:
        image: redis:6
        healthcheck:
            test: redis-cli ping
            interval: 5s
            timeout: 1s
            retries: 5
    langgraph-postgres:
        image: postgres:16
        ports:
            - "5433:5432"
        environment:
            POSTGRES_DB: postgres
            POSTGRES_USER: postgres
            POSTGRES_PASSWORD: postgres
        volumes:
            - langgraph-data:/var/lib/postgresql/data
        healthcheck:
            test: pg_isready -U postgres
            start_period: 10s
            timeout: 1s
            retries: 5
            interval: 5s
    langgraph-api:
        image: ${IMAGE_NAME}
        ports:
            - "8123:8000"
        depends_on:
            langgraph-redis:
                condition: service_healthy
            langgraph-postgres:
                condition: service_healthy
        env_file:
            - .env
        environment:
            REDIS_URI: redis://langgraph-redis:6379
            LANGSMITH_API_KEY: ${LANGSMITH_API_KEY}
            POSTGRES_URI: postgres://postgres:postgres@langgraph-postgres:5432/postgres?sslmode=disable
```

You can then run `docker compose up` with this Docker compose file in the same folder.

This will spin up LangGraph Deploy on port `8123` (if you want to change this, you can change this by changing the ports in the `langgraph-api` volume).

You can test that the application is up by checking:

```shell
curl --request GET --url 0.0.0.0:8123/ok
```
Assuming everything is running correctly, you should see a response like:

```shell
{"ok":true}
```

