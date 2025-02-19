# LangGraph Platform: Scalability & Resilience

LangGraph Platform is designed to scale horizontally with your workload. Each instance of the service is stateless, and keeps no resources in memory. The service is designed to gracefully handle new instances being added or removed, including hard shutdown cases.

## Server scalability

As you add more instances to a service, they will share the HTTP load as long as an appropriate load balancer mechanism is placed in front of them. In most deployment modalities we configure a load balancer for the service automatically. In the “self-hosted without control plane” modality it’s your responsibility to add a load balancer. Since the instances are stateless any load balancing strategy will work, no session stickiness is needed, or recommended. Any instance of the server can communicate with any queue instance (through Redis PubSub), meaning that requests to cancel or stream an in-progress run can be handled by any arbitrary instance.

## Queue scalability

As you add more instances to a service, they will increase run throughput linearly, as each instance is configured to handle a set number of concurrent runs (by default 10). Each attempt for each run will be handled by a single instance, with exactly-once semantics enforced through Postgres’s MVCC model (refer to section below for crash resilience details). Attempts that fail due to transient database errors are retried up to 3 times. We do not make use of long-lived transactions or locks, this enables us to make more efficient use of Postgres resources.

## Resilience

While a run is being handled by a queue instance, a periodic heartbeat timestamp will be recorded in Redis by that queue worker.

When a graceful shutdown request is received (SIGINT) an instance enters shutdown mode, which

- stops accepting new HTTP requests
- gives any in-progress runs a limited number of seconds to finish (if not finished it will be put back in the queue)
- stops the instance from picking up more runs from the queue

If a hard shutdown occurs, eg. due to a server crash, or an infra failure, any runs that were in progress will be picked up by a periodic sweeper task that looks for in-progress runs that have breached their heartbeat window, which will put them back in the queue for another instance to pick them up.

## Postgres resilience

For deployment modalities where we manage the Postgres database we have periodic backups, continuously replicated standby replicas for automatic failover. Optionally, on request, we can also setup read replicas as well as other advanced failover capabilities.

All communication with Postgres implements retries for retry-able errors. If Postgres is momentarily unavailable, such as during a database restart, most/all traffic should continue to succeed. Prolonged failure of the Postgres instance will switch traffic to the failover replica. If the failover replica also fails before the primary is brought back online the service would become unavailable.

## Redis resilience

All data that requires durable storage is stored in Postgres, not Redis. Redis is used only for ephemeral metadata, and communication between instances. Refer to the [architecture](./platform_architecture.md) page for more details on how we use Redis. Therefore we place no durability requirements on Redis.

All communication with Redis implements retries for retry-able errors. If Redis is momentarily unavailable, such as during a database restart, most/all traffic should continue to succeed. Prolonged failure of Redis will render the LGP service unavailable.
