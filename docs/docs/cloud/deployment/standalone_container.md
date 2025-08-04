# How to Deploy a Standalone Container

Before deploying, review the [conceptual guide for the Standalone Container](../../concepts/langgraph_standalone_container.md) deployment option.

## Prerequisites

1. Use the [LangGraph CLI](../../concepts/langgraph_cli.md) to [test your application locally](../../tutorials/langgraph-platform/local-server.md).
1. Use the [LangGraph CLI](../../concepts/langgraph_cli.md) to build a Docker image (i.e. `langgraph build`).
1. The following environment variables are needed for a standalone container deployment.
    1. `REDIS_URI`: Connection details to a Redis instance. Redis will be used as a pub-sub broker to enable streaming real time output from background runs. The value of `REDIS_URI` must be a valid [Redis connection URI](https://redis-py.readthedocs.io/en/stable/connections.html#redis.Redis.from_url).

        !!! Note "Shared Redis Instance"
            Multiple self-hosted deployments can share the same Redis instance. For example, for `Deployment A`, `REDIS_URI` can be set to `redis://<hostname_1>:<port>/1` and for `Deployment B`, `REDIS_URI` can be set to `redis://<hostname_1>:<port>/2`.

            `1` and `2` are different database numbers within the same instance, but `<hostname_1>` is shared. **The same database number cannot be used for separate deployments**.

    1. `DATABASE_URI`: Postgres connection details. Postgres will be used to store assistants, threads, runs, persist thread state and long term memory, and to manage the state of the background task queue with 'exactly once' semantics. The value of `DATABASE_URI` must be a valid [Postgres connection URI](https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING-URIS).

        !!! Note "Shared Postgres Instance"
            Multiple self-hosted deployments can share the same Postgres instance. For example, for `Deployment A`, `DATABASE_URI` can be set to `postgres://<user>:<password>@/<database_name_1>?host=<hostname_1>` and for `Deployment B`, `DATABASE_URI` can be set to `postgres://<user>:<password>@/<database_name_2>?host=<hostname_1>`.

            `<database_name_1>` and `database_name_2` are different databases within the same instance, but `<hostname_1>` is shared. **The same database cannot be used for separate deployments**.

    1. `LANGGRAPH_CLOUD_LICENSE_KEY`: (if using [Enterprise](../../concepts/langgraph_data_plane.md#licensing)) LangGraph Platform license key. This will be used to authenticate ONCE at server start up.
    1. `LANGSMITH_ENDPOINT`: To send traces to a [self-hosted LangSmith](https://docs.smith.langchain.com/self_hosting) instance, set `LANGSMITH_ENDPOINT` to the hostname of the self-hosted LangSmith instance.
1. Egress to `https://beacon.langchain.com` from your network. This is required for license verification and usage reporting if not running in air-gapped mode. See the [Egress documentation](../../cloud/deployment/egress.md) for more details.

## Kubernetes (Helm)

Use this [Helm chart](https://github.com/langchain-ai/helm/blob/main/charts/langgraph-cloud/README.md) to deploy a LangGraph Server to a Kubernetes cluster.

## Docker

Run the following `docker` command:
```shell
docker run \
    --env-file .env \
    -p 8123:8000 \
    -e REDIS_URI="foo" \
    -e DATABASE_URI="bar" \
    -e LANGSMITH_API_KEY="baz" \
    my-image
```

!!! note

    * You need to replace `my-image` with the name of the image you built in the prerequisite steps (from `langgraph build`)
    and you should provide appropriate values for `REDIS_URI`, `DATABASE_URI`, and `LANGSMITH_API_KEY`.
    * If your application requires additional environment variables, you can pass them in a similar way.

## Docker Compose

Docker Compose YAML file:
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

You can run the command `docker compose up` with this Docker Compose file in the same folder.

This will launch a LangGraph Server on port `8123` (if you want to change this, you can change this by changing the ports in the `langgraph-api` volume). You can test if the application is healthy by running:

```shell
curl --request GET --url 0.0.0.0:8123/ok
```
Assuming everything is running correctly, you should see a response like:

```shell
{"ok":true}
```
