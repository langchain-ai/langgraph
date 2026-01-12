from typing import Any, Literal, TypedDict

Distros = Literal["debian", "wolfi", "bullseye", "bookworm"]
MiddlewareOrders = Literal["auth_first", "middleware_first"]


class TTLConfig(TypedDict, total=False):
    """Configuration for TTL (time-to-live) behavior in the store."""

    refresh_on_read: bool
    """Default behavior for refreshing TTLs on read operations (`GET` and `SEARCH`).
    
    If `True`, TTLs will be refreshed on read operations (get/search) by default.
    This can be overridden per-operation by explicitly setting `refresh_ttl`.
    Defaults to `True` if not configured.
    """
    default_ttl: float | None
    """Optional. Default TTL (time-to-live) in minutes for new items.
    
    If provided, all new items will have this TTL unless explicitly overridden.
    If omitted, items will have no TTL by default.
    """
    sweep_interval_minutes: int | None
    """Optional. Interval in minutes between TTL sweep iterations.
    
    If provided, the store will periodically delete expired items based on the TTL.
    If omitted, no automatic sweeping will occur.
    """


class IndexConfig(TypedDict, total=False):
    """Configuration for indexing documents for semantic search in the store.

    This governs how text is converted into embeddings and stored for vector-based lookups.
    """

    dims: int
    """Required. Dimensionality of the embedding vectors you will store.
    
    Must match the output dimension of your selected embedding model or custom embed function.
    If mismatched, you will likely encounter shape/size errors when inserting or querying vectors.
    
    Common embedding model output dimensions:
        - openai:text-embedding-3-large: 3072
        - openai:text-embedding-3-small: 1536
        - openai:text-embedding-ada-002: 1536
        - cohere:embed-english-v3.0: 1024
        - cohere:embed-english-light-v3.0: 384
        - cohere:embed-multilingual-v3.0: 1024
        - cohere:embed-multilingual-light-v3.0: 384
    """

    embed: str
    """Required. Identifier or reference to the embedding model or a custom embedding function.
    
    The format can vary:
      - "<provider>:<model_name>" for recognized providers (e.g., "openai:text-embedding-3-large")
      - "path/to/module.py:function_name" for your own local embedding function
      - "my_custom_embed" if it's a known alias in your system

     Examples:
        - "openai:text-embedding-3-large"
        - "cohere:embed-multilingual-v3.0"
        - "src/app.py:embeddings"
    
    Note: Must return embeddings of dimension `dims`.
    """

    fields: list[str] | None
    """Optional. List of JSON fields to extract before generating embeddings.
    
    Defaults to ["$"], which means the entire JSON object is embedded as one piece of text.
    If you provide multiple fields (e.g. ["title", "content"]), each is extracted and embedded separately,
    often saving token usage if you only care about certain parts of the data.
    
    Example:
        fields=["title", "abstract", "author.biography"]
    """


class StoreConfig(TypedDict, total=False):
    """Configuration for the built-in long-term memory store.

    This store can optionally perform semantic search. If you omit `index`,
    the store will just handle traditional (non-embedded) data without vector lookups.
    """

    index: IndexConfig | None
    """Optional. Defines the vector-based semantic search configuration.
    
    If provided, the store will:
      - Generate embeddings according to `index.embed`
      - Enforce the embedding dimension given by `index.dims`
      - Embed only specified JSON fields (if any) from `index.fields`
    
    If omitted, no vector index is initialized.
    """

    ttl: TTLConfig | None
    """Optional. Defines the TTL (time-to-live) behavior configuration.
    
    If provided, the store will apply TTL settings according to the configuration.
    If omitted, no TTL behavior is configured.
    """


class ThreadTTLConfig(TypedDict, total=False):
    """Configure a default TTL for checkpointed data within threads."""

    strategy: Literal["delete"]
    """Strategy to use for deleting checkpointed data.
    
    Choices:
      - "delete": Delete all checkpoints for a thread after TTL expires.
    """
    default_ttl: float | None
    """Default TTL (time-to-live) in minutes for checkpointed data."""
    sweep_interval_minutes: int | None
    """Interval in minutes between sweep iterations.
    If omitted, a default interval will be used (typically ~ 5 minutes)."""


class SerdeConfig(TypedDict, total=False):
    """Configuration for the built-in serde, which handles checkpointing of state.

    If omitted, no serde is set up (the object store will still be present, however)."""

    allowed_json_modules: list[list[str]] | bool | None
    """Optional. List of allowed python modules to de-serialize custom objects from.
    
    If provided, only the specified modules will be allowed to be deserialized.
    If omitted, no modules are allowed, and the object returned will simply be a json object OR
    a deserialized langchain object.
    
    Example:
    {...
        "serde": {
            "allowed_json_modules": [
                ["my_agent", "my_file", "SomeType"],
            ]
        }
    }

    If you set this to True, any module will be allowed to be deserialized.

    Example:
    {...
        "serde": {
            "allowed_json_modules": true
        }
    }
    
    """
    pickle_fallback: bool
    """Optional. Whether to allow pickling as a fallback for deserialization.
    
    If True, pickling will be allowed as a fallback for deserialization.
    If False, pickling will not be allowed as a fallback for deserialization.
    Defaults to True if not configured."""


class CheckpointerConfig(TypedDict, total=False):
    """Configuration for the built-in checkpointer, which handles checkpointing of state.

    If omitted, no checkpointer is set up (the object store will still be present, however).
    """

    ttl: ThreadTTLConfig | None
    """Optional. Defines the TTL (time-to-live) behavior configuration.
    
    If provided, the checkpointer will apply TTL settings according to the configuration.
    If omitted, no TTL behavior is configured.
    """
    serde: SerdeConfig | None
    """Optional. Defines the serde configuration.
    
    If provided, the checkpointer will apply serde settings according to the configuration.
    If omitted, no serde behavior is configured.

    This configuration requires server version 0.5 or later to take effect.
    """


class SecurityConfig(TypedDict, total=False):
    """Configuration for OpenAPI security definitions and requirements.

    Useful for specifying global or path-level authentication and authorization flows
    (e.g., OAuth2, API key headers, etc.).
    """

    securitySchemes: dict[str, dict[str, Any]]
    """Describe each security scheme recognized by your OpenAPI spec.
    
    Keys are scheme names (e.g. "OAuth2", "ApiKeyAuth") and values are their definitions.
    Example:
        {
            "OAuth2": {
                "type": "oauth2",
                "flows": {
                    "password": {
                        "tokenUrl": "/token",
                        "scopes": {"read": "Read data", "write": "Write data"}
                    }
                }
            }
        }
    """
    security: list[dict[str, list[str]]]
    """Global security requirements across all endpoints.
    
    Each element in the list maps a security scheme (e.g. "OAuth2") to a list of scopes (e.g. ["read", "write"]).
    Example:
        [
            {"OAuth2": ["read", "write"]},
            {"ApiKeyAuth": []}
        ]
    """
    # path => {method => security}
    paths: dict[str, dict[str, list[dict[str, list[str]]]]]
    """Path-specific security overrides.
    
    Keys are path templates (e.g., "/items/{item_id}"), mapping to:
      - Keys that are HTTP methods (e.g., "GET", "POST"),
      - Values are lists of security definitions (just like `security`) for that method.
    
    Example:
        {
            "/private_data": {
                "GET": [{"OAuth2": ["read"]}],
                "POST": [{"OAuth2": ["write"]}]
            }
        }
    """


class CacheConfig(TypedDict, total=False):
    cache_keys: list[str]
    """Optional. List of header keys to use for caching.
    
    Example:
        ["user_id", "workspace_id"]
    """
    ttl_seconds: int
    """Optional. Time-to-live in seconds for cached items.
    
    Example:
        3600
    """
    max_size: int
    """Optional. Maximum size of the cache.
    
    Example:
        100
    """


class AuthConfig(TypedDict, total=False):
    """Configuration for custom authentication logic and how it integrates into the OpenAPI spec."""

    path: str
    """Required. Path to an instance of the Auth() class that implements custom authentication.
    
    Format: "path/to/file.py:my_auth"
    """
    disable_studio_auth: bool
    """Optional. Whether to disable LangSmith API-key authentication for requests originating the Studio. 
    
    Defaults to False, meaning that if a particular header is set, the server will verify the `x-api-key` header
    value is a valid API key for the deployment's workspace. If `True`, all requests will go through your custom
    authentication logic, regardless of origin of the request.
    """
    openapi: SecurityConfig
    """The security configuration to include in your server's OpenAPI spec.
    
    Example (OAuth2):
        {
            "securitySchemes": {
                "OAuth2": {
                    "type": "oauth2",
                    "flows": {
                        "password": {
                            "tokenUrl": "/token",
                            "scopes": {"me": "Read user info", "items": "Manage items"}
                        }
                    }
                }
            },
            "security": [
                {"OAuth2": ["me"]}
            ]
        }
    """
    cache: CacheConfig
    """Optional. Cache configuration for the server.
    
    Example:
        {
            "cache_keys": ["user_id", "workspace_id"],
            "ttl_seconds": 3600,
            "max_size": 100
        }
    """


class EncryptionConfig(TypedDict, total=False):
    """Configuration for custom at-rest encryption logic.

    Allows you to implement custom encryption for sensitive data stored in the database,
    including metadata fields and checkpoint blobs.
    """

    path: str
    """Required. Path to an instance of the Encryption() class that implements custom encryption handlers.

    Format: "path/to/file.py:my_encryption"

    Example:
        {
            "encryption": {
                "path": "./encryption.py:my_encryption"
            }
        }
    """


class CorsConfig(TypedDict, total=False):
    """Specifies Cross-Origin Resource Sharing (CORS) rules for your server.

    If omitted, defaults are typically very restrictive (often no cross-origin requests).
    Configure carefully if you want to allow usage from browsers hosted on other domains.
    """

    allow_origins: list[str]
    """Optional. List of allowed origins (e.g., "https://example.com").
    
    Default is often an empty list (no external origins). 
    Use "*" only if you trust all origins, as that bypasses most restrictions.
    """
    allow_methods: list[str]
    """Optional. HTTP methods permitted for cross-origin requests (e.g. ["GET", "POST"]).
    
    Default might be ["GET", "POST", "OPTIONS"] depending on your server framework.
    """
    allow_headers: list[str]
    """Optional. HTTP headers that can be used in cross-origin requests (e.g. ["Content-Type", "Authorization"])."""
    allow_credentials: bool
    """Optional. If `True`, cross-origin requests can include credentials (cookies, auth headers).
    
    Default False to avoid accidentally exposing secured endpoints to untrusted sites.
    """
    allow_origin_regex: str
    """Optional. A regex pattern for matching allowed origins, used if you have dynamic subdomains.
    
    Example: "^https://.*\\.mycompany\\.com$"
    """
    expose_headers: list[str]
    """Optional. List of headers that browsers are allowed to read from the response in cross-origin contexts."""
    max_age: int
    """Optional. How many seconds the browser may cache preflight responses.
    
    Default might be 600 (10 minutes). Larger values reduce preflight requests but can cause stale configurations.
    """


class ConfigurableHeaderConfig(TypedDict, total=False):
    """Customize which headers to include as configurable values in your runs.

    By default, omits x-api-key, x-tenant-id, and x-service-key.

    Exclusions (if provided) take precedence.

    Each value can be a raw string with an optional wildcard.
    """

    includes: list[str] | None
    """Headers to include (if not also matched against an 'excludes' pattern).

    Examples:
        - 'user-agent'
        - 'x-configurable-*'
    """
    excludes: list[str] | None
    """Headers to exclude. Applied before the 'includes' checks.

    Examples:
        - 'x-api-key'
        - '*key*'
        - '*token*'
    """


class HttpConfig(TypedDict, total=False):
    """Configuration for the built-in HTTP server that powers your deployment's routes and endpoints."""

    app: str
    """Optional. Import path to a custom Starlette/FastAPI application to mount.
    
    Format: "path/to/module.py:app_var"
    If provided, it can override or extend the default routes.
    """
    disable_assistants: bool
    """Optional. If `True`, /assistants routes are removed from the server.
    
    Default is False (meaning /assistants is enabled).
    """
    disable_threads: bool
    """Optional. If `True`, /threads routes are removed.
    
    Default is False.
    """
    disable_runs: bool
    """Optional. If `True`, /runs routes are removed.
    
    Default is False.
    """
    disable_store: bool
    """Optional. If `True`, /store routes are removed, disabling direct store interactions via HTTP.
    
    Default is False.
    """
    disable_mcp: bool
    """Optional. If `True`, /mcp routes are removed, disabling default support to expose the deployment as an MCP server.
    
    Default is False.
    """
    disable_a2a: bool
    """Optional. If `True`, /a2a routes are removed, disabling default support to expose the deployment as an agent-to-agent (A2A) server.
    
    Default is False.
    """
    disable_meta: bool
    """Optional. Remove meta endpoints.
    
    Set to True to disable the following endpoints: /openapi.json, /info, /metrics, /docs.
    This will also make the /ok endpoint skip any DB or other checks, always returning {"ok": True}.
    
    Default is False.
    """
    disable_ui: bool
    """Optional. If `True`, /ui routes are removed, disabling the UI server.
    
    Default is False.
    """
    disable_webhooks: bool
    """Optional. If `True`, webhooks are disabled. Runs created with an associated webhook will
    still be executed, but the webhook event will not be sent.
    
    Default is False.
    """
    cors: CorsConfig | None
    """Optional. Defines CORS restrictions. If omitted, no special rules are set and 
    cross-origin behavior depends on default server settings.
    """
    configurable_headers: ConfigurableHeaderConfig | None
    """Optional. Defines how headers are treated for a run's configuration.

    You can include or exclude headers as configurable values to condition your
    agent's behavior or permissions on a request's headers."""
    logging_headers: ConfigurableHeaderConfig | None
    """Optional. Defines which headers are excluded from logging."""
    middleware_order: MiddlewareOrders | None
    """Optional. Defines the order in which to apply server customizations.

    Choices:
      - "auth_first": Authentication hooks (custom or default) are evaluated
      before custom middleware.
      - "middleware_first": Custom middleware is evaluated
      before authentication hooks (custom or default).

    Default is `middleware_first`.
    """
    enable_custom_route_auth: bool
    """Optional. If `True`, authentication is enabled for custom routes,
    not just the routes that are protected by default.
    (Routes protected by default include /assistants, /threads, and /runs).

    Default is False. This flag only affects authentication behavior
    if `app` is provided and contains custom routes.
    """
    mount_prefix: str
    """Optional. URL prefix to prepend to all the routes.
    
    Example:
        "/api"
    """


class WebhookUrlPolicy(TypedDict, total=False):
    require_https: bool
    """Enforce HTTPS scheme for absolute URLs; reject `http://` when true."""
    allowed_domains: list[str]
    """Hostname allowlist. Supports exact hosts and wildcard subdomains.

    Use entries like "hooks.example.com" or "*.mycorp.com". The wildcard only
    matches subdomains ("foo.mycorp.com"), not the apex ("mycorp.com"). When
    empty or omitted, any public host is allowed (subject to SSRF IP checks).
    """
    allowed_ports: list[int]
    """Explicit port allowlist for absolute URLs.

    If set, requests must use one of these ports. Defaults are respected when
    a port is not present in the URL (443 for https, 80 for http).
    """
    max_url_length: int
    """Maximum permitted URL length in characters; longer inputs are rejected early."""
    disable_loopback: bool
    """Disallow relative URLs (internal loopback calls) when true."""


class WebhooksConfig(TypedDict, total=False):
    env_prefix: str
    """Required prefix for environment variables referenced in header templates.

    Acts as an allowlist boundary to prevent leaking arbitrary environment
    variables. Defaults to "LG_WEBHOOK_" when omitted.
    """
    url: WebhookUrlPolicy
    """URL validation policy for user-supplied webhook endpoints."""
    headers: dict[str, str]
    """Static headers to include with webhook requests.

    Values may contain templates of the form "${{ env.VAR }}". On startup, these
    are resolved via the process environment after verifying `VAR` starts with
    `env_prefix`. Mixed literals and multiple templates are allowed.
    """


class Config(TypedDict, total=False):
    """Top-level config for langgraph-cli or similar deployment tooling."""

    python_version: str
    """Optional. Python version in 'major.minor' format (e.g. '3.11'). 
    Must be at least 3.11 or greater for this deployment to function properly.
    """

    node_version: str | None
    """Optional. Node.js version as a major version (e.g. '20'), if your deployment needs Node.
    Must be >= 20 if provided.
    """

    api_version: str | None
    """Optional. Which semantic version of the LangGraph API server to use.
    
    Defaults to latest. Check the
    [changelog](https://docs.langchain.com/langgraph-platform/langgraph-server-changelog)
    for more information."""

    _INTERNAL_docker_tag: str | None
    """Optional. Internal use only.
    """

    base_image: str | None
    """Optional. Base image to use for the LangGraph API server.
    
    Defaults to langchain/langgraph-api or langchain/langgraphjs-api."""

    image_distro: Distros | None
    """Optional. Linux distribution for the base image.
    
    Must be one of 'wolfi', 'debian', 'bullseye', or 'bookworm'.
    If omitted, defaults to 'debian' ('latest').
    """

    pip_config_file: str | None
    """Optional. Path to a pip config file (e.g., "/etc/pip.conf" or "pip.ini") for controlling
    package installation (custom indices, credentials, etc.).
    
    Only relevant if Python dependencies are installed via pip. If omitted, default pip settings are used.
    """

    pip_installer: str | None
    """Optional. Python package installer to use ('auto', 'pip', 'uv').
    
    - 'auto' (default): Use uv for supported base images, otherwise pip
    - 'pip': Force use of pip regardless of base image support
    - 'uv': Force use of uv (will fail if base image doesn't support it)
    """

    dockerfile_lines: list[str]
    """Optional. Additional Docker instructions that will be appended to your base Dockerfile.
    
    Useful for installing OS packages, setting environment variables, etc. 
    Example:
        dockerfile_lines=[
            "RUN apt-get update && apt-get install -y libmagic-dev",
            "ENV MY_CUSTOM_VAR=hello_world"
        ]
    """

    dependencies: list[str]
    """List of Python dependencies to install, either from PyPI or local paths.
    
    Examples:
      - "." or "./src" if you have a local Python package
      - str (aka "anthropic") for a PyPI package
      - "git+https://github.com/org/repo.git@main" for a Git-based package
    Defaults to an empty list, meaning no additional packages installed beyond your base environment.
    """

    graphs: dict[str, str]
    """Optional. Named definitions of graphs, each pointing to a Python object.

    
    Graphs can be StateGraph, @entrypoint, or any other Pregel object OR they can point to (async) context
    managers that accept a single configuration argument (of type RunnableConfig) and return a pregel object
    (instance of Stategraph, etc.).
    
    Keys are graph names, values are "path/to/file.py:object_name".
    Example:
        {
            "mygraph": "graphs/my_graph.py:graph_definition",
            "anothergraph": "graphs/another.py:get_graph"
        }
    """

    env: dict[str, str] | str
    """Optional. Environment variables to set for your deployment.
    
    - If given as a dict, keys are variable names and values are their values.
    - If given as a string, it must be a path to a file containing lines in KEY=VALUE format.
    
    Example as a dict:
        env={"API_TOKEN": "abc123", "DEBUG": "true"}
    Example as a file path:
        env=".env"
    """

    store: StoreConfig | None
    """Optional. Configuration for the built-in long-term memory store, including semantic search indexing.
    
    If omitted, no vector index is set up (the object store will still be present, however).
    """

    checkpointer: CheckpointerConfig | None
    """Optional. Configuration for the built-in checkpointer, which handles checkpointing of state.
    
    If omitted, no checkpointer is set up (the object store will still be present, however).
    """

    auth: AuthConfig | None
    """Optional. Custom authentication config, including the path to your Python auth logic and
    the OpenAPI security definitions it uses.
    """

    encryption: EncryptionConfig | None
    """Optional. Custom at-rest encryption config, including the path to your Python encryption logic.

    Allows you to implement custom encryption for sensitive data stored in the database.
    """

    http: HttpConfig | None
    """Optional. Configuration for the built-in HTTP server, controlling which custom routes are exposed
    and how cross-origin requests are handled.
    """

    webhooks: WebhooksConfig | None
    """Optional. Webhooks configuration for outbound event delivery.

    Forwarded into the container as `LANGGRAPH_WEBHOOKS`. See `WebhooksConfig`
    for URL policy and header templating details.
    """

    ui: dict[str, str] | None
    """Optional. Named definitions of UI components emitted by the agent, each pointing to a JS/TS file.
    """

    keep_pkg_tools: bool | list[str] | None
    """Optional. Control whether to retain Python packaging tools in the final image.
    
    Allowed tools are: "pip", "setuptools", "wheel".
    You can also set to true to include all packaging tools.
    """


__all__ = [
    "Config",
    "StoreConfig",
    "CheckpointerConfig",
    "AuthConfig",
    "EncryptionConfig",
    "HttpConfig",
    "MiddlewareOrders",
    "Distros",
    "TTLConfig",
    "IndexConfig",
]
