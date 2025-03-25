import json
import os
import pathlib
import textwrap
from collections import Counter
from typing import Any, NamedTuple, Optional, TypedDict, Union

import click

MIN_NODE_VERSION = "20"
MIN_PYTHON_VERSION = "3.11"


class TTLConfig(TypedDict, total=False):
    """Configuration for TTL (time-to-live) behavior in the store."""

    refresh_on_read: bool
    """Default behavior for refreshing TTLs on read operations (GET and SEARCH).
    
    If True, TTLs will be refreshed on read operations (get/search) by default.
    This can be overridden per-operation by explicitly setting refresh_ttl.
    Defaults to True if not configured.
    """
    default_ttl: Optional[float]
    """Optional. Default TTL (time-to-live) in minutes for new items.
    
    If provided, all new items will have this TTL unless explicitly overridden.
    If omitted, items will have no TTL by default.
    """
    sweep_interval_minutes: Optional[int]
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

    fields: Optional[list[str]]
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

    index: Optional[IndexConfig]
    """Optional. Defines the vector-based semantic search configuration.
    
    If provided, the store will:
      - Generate embeddings according to `index.embed`
      - Enforce the embedding dimension given by `index.dims`
      - Embed only specified JSON fields (if any) from `index.fields`
    
    If omitted, no vector index is initialized.
    """

    ttl: Optional[TTLConfig]
    """Optional. Defines the TTL (time-to-live) behavior configuration.
    
    If provided, the store will apply TTL settings according to the configuration.
    If omitted, no TTL behavior is configured.
    """


class SecurityConfig(TypedDict, total=False):
    """Configuration for OpenAPI security definitions and requirements.

    Useful for specifying global or path-level authentication and authorization flows
    (e.g., OAuth2, API key headers, etc.).
    """

    securitySchemes: dict[str, dict[str, Any]]
    """Required. Dict describing each security scheme recognized by your OpenAPI spec.
    
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
    """Optional. Global security requirements across all endpoints.
    
    Each element in the list maps a security scheme (e.g. "OAuth2") to a list of scopes (e.g. ["read", "write"]).
    Example:
        [
            {"OAuth2": ["read", "write"]},
            {"ApiKeyAuth": []}
        ]
    """
    # path => {method => security}
    paths: dict[str, dict[str, list[dict[str, list[str]]]]]
    """Optional. Path-specific security overrides.
    
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


class AuthConfig(TypedDict, total=False):
    """Configuration for custom authentication logic and how it integrates into the OpenAPI spec."""

    path: str
    """Required. Path to an instance of the Auth() class that implements custom authentication.
    
    Format: "path/to/file.py:my_auth"
    """
    disable_studio_auth: bool
    """Optional. Whether to disable LangSmith API-key authentication for requests originating the Studio. 
    
    Defaults to False, meaning that if a particular header is set, the server will verify the `x-api-key` header
    value is a valid API key for the deployment's workspace. If True, all requests will go through your custom
    authentication logic, regardless of origin of the request.
    """
    openapi: SecurityConfig
    """Required. Detailed security configuration that merges into your deployment's OpenAPI spec.
    
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
    """Optional. If True, cross-origin requests can include credentials (cookies, auth headers).
    
    Default False to avoid accidentally exposing secured endpoints to untrusted sites.
    """
    allow_origin_regex: str
    """Optional. A regex pattern for matching allowed origins, used if you have dynamic subdomains.
    
    Example: "^https://.*\.mycompany\.com$"
    """
    expose_headers: list[str]
    """Optional. List of headers that browsers are allowed to read from the response in cross-origin contexts."""
    max_age: int
    """Optional. How many seconds the browser may cache preflight responses.
    
    Default might be 600 (10 minutes). Larger values reduce preflight requests but can cause stale configurations.
    """


class HttpConfig(TypedDict, total=False):
    """Configuration for the built-in HTTP server that powers your deployment's routes and endpoints."""

    app: str
    """Optional. Import path to a custom Starlette/FastAPI application to mount.
    
    Format: "path/to/module.py:app_var"
    If provided, it can override or extend the default routes.
    """
    disable_assistants: bool
    """Optional. If True, /assistants routes are removed from the server.
    
    Default is False (meaning /assistants is enabled).
    """
    disable_threads: bool
    """Optional. If True, /threads routes are removed.
    
    Default is False.
    """
    disable_runs: bool
    """Optional. If True, /runs routes are removed.
    
    Default is False.
    """
    disable_store: bool
    """Optional. If True, /store routes are removed, disabling direct store interactions via HTTP.
    
    Default is False.
    """
    disable_meta: bool
    """Optional. If True, all meta endpoints (/ok, /info, /metrics, /docs) are disabled.
    
    Default is False.
    """
    cors: Optional[CorsConfig]
    """Optional. Defines CORS restrictions. If omitted, no special rules are set and 
    cross-origin behavior depends on default server settings.
    """


class Config(TypedDict, total=False):
    """Top-level config for langgraph-cli or similar deployment tooling."""

    python_version: str
    """Optional. Python version in 'major.minor' format (e.g. '3.11'). 
    Must be at least 3.11 or greater for this deployment to function properly.
    """

    node_version: Optional[str]
    """Optional. Node.js version as a major version (e.g. '20'), if your deployment needs Node.
    Must be >= 20 if provided.
    """

    pip_config_file: Optional[str]
    """Optional. Path to a pip config file (e.g., "/etc/pip.conf" or "pip.ini") for controlling
    package installation (custom indices, credentials, etc.).
    
    Only relevant if Python dependencies are installed via pip. If omitted, default pip settings are used.
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

    env: Union[dict[str, str], str]
    """Optional. Environment variables to set for your deployment.
    
    - If given as a dict, keys are variable names and values are their values.
    - If given as a string, it must be a path to a file containing lines in KEY=VALUE format.
    
    Example as a dict:
        env={"API_TOKEN": "abc123", "DEBUG": "true"}
    Example as a file path:
        env=".env"
    """

    store: Optional[StoreConfig]
    """Optional. Configuration for the built-in long-term memory store, including semantic search indexing.
    
    If omitted, no vector index is set up (the object store will still be present, however).
    """

    auth: Optional[AuthConfig]
    """Optional. Custom authentication config, including the path to your Python auth logic and 
    the OpenAPI security definitions it uses.
    """

    http: Optional[HttpConfig]
    """Optional. Configuration for the built-in HTTP server, controlling which custom routes are exposed
    and how cross-origin requests are handled.
    """

    ui: Optional[dict[str, str]]
    """Optional. Named definitions of UI components emitted by the agent, each pointing to a JS/TS file.
    """


def _parse_version(version_str: str) -> tuple[int, int]:
    """Parse a version string into a tuple of (major, minor)."""
    try:
        major, minor = map(int, version_str.split("-")[0].split("."))
        return (major, minor)
    except ValueError:
        raise click.UsageError(f"Invalid version format: {version_str}") from None


def _parse_node_version(version_str: str) -> int:
    """Parse a Node.js version string into a major version number."""
    try:
        if "." in version_str:
            raise ValueError("Node.js version must be major version only")
        return int(version_str)
    except ValueError:
        raise click.UsageError(
            f"Invalid Node.js version format: {version_str}. "
            "Use major version only (e.g., '20')."
        ) from None


def validate_config(config: Config) -> Config:
    """Validate a configuration dictionary."""
    config = (
        {
            "node_version": config.get("node_version"),
            "dockerfile_lines": config.get("dockerfile_lines", []),
            "dependencies": config.get("dependencies", []),
            "graphs": config.get("graphs", {}),
            "env": config.get("env", {}),
            "store": config.get("store"),
            "auth": config.get("auth"),
            "http": config.get("http"),
            "ui": config.get("ui"),
        }
        if config.get("node_version")
        else {
            "python_version": config.get("python_version", "3.11"),
            "pip_config_file": config.get("pip_config_file"),
            "dockerfile_lines": config.get("dockerfile_lines", []),
            "dependencies": config.get("dependencies", []),
            "graphs": config.get("graphs", {}),
            "env": config.get("env", {}),
            "store": config.get("store"),
            "auth": config.get("auth"),
            "http": config.get("http"),
            "ui": config.get("ui"),
        }
    )

    if config.get("node_version"):
        node_version = config["node_version"]
        try:
            major = _parse_node_version(node_version)
            min_major = _parse_node_version(MIN_NODE_VERSION)
            if major < min_major:
                raise click.UsageError(
                    f"Node.js version {node_version} is not supported. "
                    f"Minimum required version is {MIN_NODE_VERSION}."
                )
        except ValueError as e:
            raise click.UsageError(str(e)) from None

    if config.get("python_version"):
        pyversion = config["python_version"]
        if not pyversion.count(".") == 1 or not all(
            part.isdigit() for part in pyversion.split("-")[0].split(".")
        ):
            raise click.UsageError(
                f"Invalid Python version format: {pyversion}. "
                "Use 'major.minor' format (e.g., '3.11'). "
                "Patch version cannot be specified."
            )
        if _parse_version(pyversion) < _parse_version(MIN_PYTHON_VERSION):
            raise click.UsageError(
                f"Python version {pyversion} is not supported. "
                f"Minimum required version is {MIN_PYTHON_VERSION}."
            )

        if not config["dependencies"]:
            raise click.UsageError(
                "No dependencies found in config. "
                "Add at least one dependency to 'dependencies' list."
            )

    if not config["graphs"]:
        raise click.UsageError(
            "No graphs found in config. "
            "Add at least one graph to 'graphs' dictionary."
        )

    # Validate auth config
    if auth_conf := config.get("auth"):
        if "path" in auth_conf:
            if ":" not in auth_conf["path"]:
                raise ValueError(
                    f"Invalid auth.path format: '{auth_conf['path']}'. "
                    "Must be in format './path/to/file.py:attribute_name'"
                )
    if http_conf := config.get("http"):
        if "app" in http_conf:
            if ":" not in http_conf["app"]:
                raise ValueError(
                    f"Invalid http.app format: '{http_conf['app']}'. "
                    "Must be in format './path/to/file.py:attribute_name'"
                )
    return config


def validate_config_file(config_path: pathlib.Path) -> Config:
    """Load and validate a configuration file."""
    with open(config_path) as f:
        config = json.load(f)
    validated = validate_config(config)
    # Enforce the package.json doesn't enforce an
    # incompatible Node.js version
    if validated.get("node_version"):
        package_json_path = config_path.parent / "package.json"
        if package_json_path.is_file():
            try:
                with open(package_json_path) as f:
                    package_json = json.load(f)
                    if "engines" in package_json:
                        engines = package_json["engines"]
                        if any(engine != "node" for engine in engines.keys()):
                            raise click.UsageError(
                                "Only 'node' engine is supported in package.json engines."
                                f" Got engines: {list(engines.keys())}"
                            )
                        if engines:
                            node_version = engines["node"]
                            try:
                                major = _parse_node_version(node_version)
                                min_major = _parse_node_version(MIN_NODE_VERSION)
                                if major < min_major:
                                    raise click.UsageError(
                                        f"Node.js version in package.json engines must be >= {MIN_NODE_VERSION} "
                                        f"(major version only), got '{node_version}'. Minor/patch versions "
                                        "(like '20.x.y') are not supported to prevent deployment issues "
                                        "when new Node.js versions are released."
                                    )
                            except ValueError as e:
                                raise click.UsageError(str(e)) from None

            except json.JSONDecodeError:
                raise click.UsageError(
                    "Invalid package.json found in langgraph "
                    f"config directory {package_json_path}: file is not valid JSON"
                ) from None
    return validated


class LocalDeps(NamedTuple):
    """A container for referencing and managing local Python dependencies.

    A "local dependency" is any entry in the config's `dependencies` list
    that starts with "." (dot), denoting a relative path
    to a local directory containing Python code.

    For each local dependency, the system inspects its directory to
    determine how it should be installed inside the Docker container.

    Specifically, we detect:

    - **Real packages**: Directories containing a `pyproject.toml` or a `setup.py`.
      These can be installed with pip as a regular Python package.
    - **Faux packages**: Directories that do not include a `pyproject.toml` or
      `setup.py` but do contain Python files and possibly an `__init__.py`. For
      these, the code dynamically generates a minimal `pyproject.toml` in the
      Docker image so that they can still be installed with pip.
    - **Requirements files**: If a local dependency directory
      has a `requirements.txt`, it is tracked so that those dependencies
      can be installed within the Docker container before installing the local package.

    Attributes:
        pip_reqs: A list of (host_requirements_path, container_requirements_path)
            tuples. Each entry points to a local `requirements.txt` file and where
            it should be placed inside the Docker container before running `pip install`.

        real_pkgs: A dictionary mapping a local directory path (host side) to a
            tuple of (dependency_string, container_package_path). These directories
            contain the necessary files (e.g., `pyproject.toml` or `setup.py`) to be
            installed as a standard Python package with pip.

        faux_pkgs: A dictionary mapping a local directory path (host side) to a
            tuple of (dependency_string, container_package_path). For these
            directories—called "faux packages"—the code will generate a minimal
            `pyproject.toml` inside the Docker image. This ensures that pip
            recognizes them as installable packages, even though they do not
            natively include packaging metadata.

        working_dir: The path inside the Docker container to use as the working
            directory. If the local dependency `"."` is present in the config, this
            field captures the path where that dependency will appear in the
            container (e.g., `/deps/<name>` or similar). Otherwise, it may be `None`.

        additional_contexts: A list of paths to directories that contain local
            dependencies in parent directories. These directories are added to the
            Docker build context to ensure that the Dockerfile can access them.
    """

    pip_reqs: list[tuple[pathlib.Path, str]]
    real_pkgs: dict[pathlib.Path, tuple[str, str]]
    faux_pkgs: dict[pathlib.Path, tuple[str, str]]
    # if . is in dependencies, use it as working_dir
    working_dir: Optional[str] = None
    # if there are local dependencies in parent directories, use additional_contexts
    additional_contexts: list[pathlib.Path] = None


def _assemble_local_deps(config_path: pathlib.Path, config: Config) -> LocalDeps:
    config_path = config_path.resolve()
    # ensure reserved package names are not used
    reserved = {
        "src",
        "langgraph-api",
        "langgraph_api",
        "langgraph",
        "langchain-core",
        "langchain_core",
        "pydantic",
        "orjson",
        "fastapi",
        "uvicorn",
        "psycopg",
        "httpx",
        "langsmith",
    }
    counter = Counter()

    def check_reserved(name: str, ref: str):
        if name in reserved:
            raise ValueError(
                f"Package name '{name}' used in local dep '{ref}' is reserved. "
                "Rename the directory."
            )
        reserved.add(name)

    pip_reqs = []
    real_pkgs = {}
    faux_pkgs = {}
    working_dir: Optional[str] = None
    additional_contexts: list[pathlib.Path] = []

    for local_dep in config["dependencies"]:
        if not local_dep.startswith("."):
            # If the dependency is not a local path, skip it
            continue

        # Verify that the local dependency can be resolved
        # (e.g., this would raise an informative error if a user mistyped a path).
        resolved = (config_path.parent / local_dep).resolve()

        # validate local dependency
        if not resolved.exists():
            raise FileNotFoundError(f"Could not find local dependency: {resolved}")
        elif not resolved.is_dir():
            raise NotADirectoryError(
                f"Local dependency must be a directory: {resolved}"
            )
        elif resolved == config_path.parent:
            pass
        elif config_path.parent not in resolved.parents:
            additional_contexts.append(resolved)

        # Check for pyproject.toml or setup.py
        # If found, treat as a real package, if not treat as a faux package.
        # For faux packages, we'll also check for presence of requirements.txt.
        files = os.listdir(resolved)
        if "pyproject.toml" in files or "setup.py" in files:
            # real package

            # assign a unique folder name
            container_name = resolved.name
            if counter[container_name] > 0:
                container_name += f"_{counter[container_name]}"
            counter[container_name] += 1
            # add to deps
            real_pkgs[resolved] = (local_dep, container_name)
            # set working_dir
            if local_dep == ".":
                working_dir = f"/deps/{container_name}"
        else:
            # We could not find a pyproject.toml or setup.py, so treat as a faux package
            if any(file == "__init__.py" for file in files):
                # flat layout
                if "-" in resolved.name:
                    raise ValueError(
                        f"Package name '{resolved.name}' contains a hyphen. "
                        "Rename the directory to use it as flat-layout package."
                    )
                check_reserved(resolved.name, local_dep)
                container_path = f"/deps/__outer_{resolved.name}/{resolved.name}"
            else:
                # src layout
                container_path = f"/deps/__outer_{resolved.name}/src"
                for file in files:
                    rfile = resolved / file
                    if (
                        rfile.is_dir()
                        and file != "__pycache__"
                        and not file.startswith(".")
                    ):
                        try:
                            for subfile in os.listdir(rfile):
                                if subfile.endswith(".py"):
                                    check_reserved(file, local_dep)
                                    break
                        except PermissionError:
                            pass
            faux_pkgs[resolved] = (local_dep, container_path)
            if local_dep == ".":
                working_dir = container_path

            # If the faux package has a requirements.txt, we'll add
            # the path to the list of requirements to install.
            if "requirements.txt" in files:
                rfile = resolved / "requirements.txt"
                pip_reqs.append(
                    (
                        rfile,
                        f"{container_path}/requirements.txt",
                    )
                )

    return LocalDeps(pip_reqs, real_pkgs, faux_pkgs, working_dir, additional_contexts)


def _update_graph_paths(
    config_path: pathlib.Path, config: Config, local_deps: LocalDeps
) -> None:
    """Remap each graph's import path to the correct in-container path.

    The config may contain entries in `graphs` that look like this:
        {
            "my_graph": "./mygraphs/main.py:graph_function"
        }
    or
        {
            "my_graph": "./src/some_subdir/my_file.py:my_graph"
        }
    which indicate a local file (on the host) followed by a colon and a
    callable/object attribute within that file.

    During the Docker build, local directories are copied into special
    `/deps/` subdirectories, so they can be installed or referenced in
    the container. This function updates each graph's import path to
    reflect its new location **inside** the Docker container.

    Paths inside the container must be POSIX-style paths (even if
    the host system is Windows).

    Args:
        config_path: The path to the config file (e.g. `langgraph.json`).
        config: The validated configuration dictionary.
        local_deps: An object containing references to local dependencies:
            - real Python packages (with a `pyproject.toml` or `setup.py`)
            - “faux” packages that need minimal metadata to be installable
            - potential `requirements.txt` for local dependencies
            - container work directory (if "." is in `dependencies`)

    Raises:
        ValueError: If the import string is not in the format `<module>:<attribute>`
                    or if the referenced local file is not found in `dependencies`.
        FileNotFoundError: If the local file (module) does not actually exist on disk.
        IsADirectoryError: If `module_str` points to a directory instead of a file.
    """
    for graph_id, import_str in config["graphs"].items():
        module_str, _, attr_str = import_str.partition(":")
        if not module_str or not attr_str:
            message = (
                'Import string "{import_str}" must be in format "<module>:<attribute>".'
            )
            raise ValueError(message.format(import_str=import_str))

        # Check for either forward slash or backslash in the module string
        # to determine if it's a file path.
        if "/" in module_str or "\\" in module_str:
            # Resolve the local path properly on the current OS
            resolved = (config_path.parent / module_str).resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Could not find local module: {resolved}")
            elif not resolved.is_file():
                raise IsADirectoryError(f"Local module must be a file: {resolved}")
            else:
                for path in local_deps.real_pkgs:
                    if resolved.is_relative_to(path):
                        container_path = (
                            pathlib.Path("/deps")
                            / path.name
                            / resolved.relative_to(path)
                        )
                        module_str = container_path.as_posix()
                        break
                else:
                    for faux_pkg, (_, destpath) in local_deps.faux_pkgs.items():
                        if resolved.is_relative_to(faux_pkg):
                            container_subpath = resolved.relative_to(faux_pkg)
                            # Construct the final path, ensuring POSIX style
                            module_str = f"{destpath}/{container_subpath.as_posix()}"
                            break
                    else:
                        raise ValueError(
                            f"Module '{import_str}' not found in 'dependencies' list. "
                            "Add its containing package to 'dependencies' list."
                        )
            # update the config
            config["graphs"][graph_id] = f"{module_str}:{attr_str}"


def _update_auth_path(
    config_path: pathlib.Path, config: Config, local_deps: LocalDeps
) -> None:
    """Update auth.path to use Docker container paths."""
    auth_conf = config.get("auth")
    if not auth_conf or not (path_str := auth_conf.get("path")):
        return

    module_str, sep, attr_str = path_str.partition(":")
    if not sep or not module_str.startswith("."):
        return  # Already validated or absolute path

    resolved = config_path.parent / module_str
    if not resolved.exists():
        raise FileNotFoundError(f"Auth file not found: {resolved} (from {path_str})")
    if not resolved.is_file():
        raise IsADirectoryError(f"Auth path must be a file: {resolved}")

    # Check faux packages first (higher priority)
    for faux_path, (_, destpath) in local_deps.faux_pkgs.items():
        if resolved.is_relative_to(faux_path):
            new_path = f"{destpath}/{resolved.relative_to(faux_path)}:{attr_str}"
            auth_conf["path"] = new_path
            return

    # Check real packages
    for real_path in local_deps.real_pkgs:
        if resolved.is_relative_to(real_path):
            new_path = (
                f"/deps/{real_path.name}/{resolved.relative_to(real_path)}:{attr_str}"
            )
            auth_conf["path"] = new_path
            return

    raise ValueError(
        f"Auth file '{resolved}' not covered by dependencies.\n"
        "Add its parent directory to the 'dependencies' array in your config.\n"
        f"Current dependencies: {config['dependencies']}"
    )


def _update_http_app_path(
    config_path: pathlib.Path, config: Config, local_deps: LocalDeps
) -> None:
    """Update the HTTP app path to point to the correct location in the Docker container.

    Similar to _update_graph_paths, this ensures that if a custom app is specified via
    a local file path, that file is included in the Docker build context and its path
    is updated to point to the correct location in the container.
    """
    if not (http_config := config.get("http")) or not (
        app_str := http_config.get("app")
    ):
        return

    module_str, _, attr_str = app_str.partition(":")
    if not module_str or not attr_str:
        message = (
            'Import string "{import_str}" must be in format "<module>:<attribute>".'
        )
        raise ValueError(message.format(import_str=app_str))

    # Check if it's a file path
    if "/" in module_str or "\\" in module_str:
        # Resolve the local path properly on the current OS
        resolved = (config_path.parent / module_str).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Could not find HTTP app module: {resolved}")
        elif not resolved.is_file():
            raise IsADirectoryError(f"HTTP app module must be a file: {resolved}")
        else:
            for path in local_deps.real_pkgs:
                if resolved.is_relative_to(path):
                    container_path = (
                        pathlib.Path("/deps") / path.name / resolved.relative_to(path)
                    )
                    module_str = container_path.as_posix()
                    break
            else:
                for faux_pkg, (_, destpath) in local_deps.faux_pkgs.items():
                    if resolved.is_relative_to(faux_pkg):
                        container_subpath = resolved.relative_to(faux_pkg)
                        # Construct the final path, ensuring POSIX style
                        module_str = f"{destpath}/{container_subpath.as_posix()}"
                        break
                else:
                    raise ValueError(
                        f"HTTP app module '{app_str}' not found in 'dependencies' list. "
                        "Add its containing package to 'dependencies' list."
                    )
        # update the config
        http_config["app"] = f"{module_str}:{attr_str}"


def python_config_to_docker(
    config_path: pathlib.Path, config: Config, base_image: str
) -> tuple[str, dict[str, str]]:
    """Generate a Dockerfile from the configuration."""
    # configure pip
    pip_install = (
        "PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt"
    )
    if config.get("pip_config_file"):
        pip_install = f"PIP_CONFIG_FILE=/pipconfig.txt {pip_install}"
    pip_config_file_str = (
        f"ADD {config['pip_config_file']} /pipconfig.txt"
        if config.get("pip_config_file")
        else ""
    )

    # collect dependencies
    pypi_deps = [dep for dep in config["dependencies"] if not dep.startswith(".")]
    local_deps = _assemble_local_deps(config_path, config)
    # Rewrite graph paths, so they point to the correct location in the Docker container
    _update_graph_paths(config_path, config, local_deps)
    # Rewrite auth path, so it points to the correct location in the Docker container
    _update_auth_path(config_path, config, local_deps)
    # Rewrite HTTP app path, so it points to the correct location in the Docker container
    _update_http_app_path(config_path, config, local_deps)

    pip_pkgs_str = f"RUN {pip_install} {' '.join(pypi_deps)}" if pypi_deps else ""
    if local_deps.pip_reqs:
        pip_reqs_str = os.linesep.join(
            (
                f"COPY --from=__outer_{reqpath.name} requirements.txt {destpath}"
                if reqpath.parent in local_deps.additional_contexts
                else f"ADD {reqpath.relative_to(config_path.parent)} {destpath}"
            )
            for reqpath, destpath in local_deps.pip_reqs
        )
        pip_reqs_str += f'{os.linesep}RUN {pip_install} {" ".join("-r " + r for _,r in local_deps.pip_reqs)}'
        pip_reqs_str = f"""# -- Installing local requirements --
{pip_reqs_str}
# -- End of local requirements install --"""

    else:
        pip_reqs_str = ""

    # https://setuptools.pypa.io/en/latest/userguide/datafiles.html#package-data
    # https://til.simonwillison.net/python/pyproject
    faux_pkgs_str = f"{os.linesep}{os.linesep}".join(
        (
            f"""# -- Adding non-package dependency {fullpath.name} --
COPY --from=__outer_{fullpath.name} . {destpath}"""
            if fullpath in local_deps.additional_contexts
            else f"""# -- Adding non-package dependency {fullpath.name} --
ADD {relpath} {destpath}"""
        )
        + f"""
RUN set -ex && \\
    for line in '[project]' \\
                'name = "{fullpath.name}"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]'; do \\
        echo "$line" >> /deps/__outer_{fullpath.name}/pyproject.toml; \\
    done
# -- End of non-package dependency {fullpath.name} --"""
        for fullpath, (relpath, destpath) in local_deps.faux_pkgs.items()
    )

    local_pkgs_str = os.linesep.join(
        (
            f"""# -- Adding local package {relpath} --
COPY --from={name} . /deps/{name}
# -- End of local package {relpath} --"""
            if fullpath in local_deps.additional_contexts
            else f"""# -- Adding local package {relpath} --
ADD {relpath} /deps/{name}
# -- End of local package {relpath} --"""
        )
        for fullpath, (relpath, name) in local_deps.real_pkgs.items()
    )

    installs = f"{os.linesep}{os.linesep}".join(
        filter(
            None,
            [
                pip_config_file_str,
                pip_pkgs_str,
                pip_reqs_str,
                local_pkgs_str,
                faux_pkgs_str,
            ],
        )
    )

    env_vars = []

    if (store_config := config.get("store")) is not None:
        env_vars.append(f"ENV LANGGRAPH_STORE='{json.dumps(store_config)}'")

    if (auth_config := config.get("auth")) is not None:
        env_vars.append(f"ENV LANGGRAPH_AUTH='{json.dumps(auth_config)}'")

    if (http_config := config.get("http")) is not None:
        env_vars.append(f"ENV LANGGRAPH_HTTP='{json.dumps(http_config)}'")

    graphs = config["graphs"]
    env_vars.append(f"ENV LANGSERVE_GRAPHS='{json.dumps(graphs)}'")

    docker_file_contents = [
        f"FROM {base_image}:{config['python_version']}",
        "",
        os.linesep.join(config["dockerfile_lines"]),
        "",
        installs,
        "",
        "# -- Installing all local dependencies --",
        f"RUN {pip_install} -e /deps/*",
        "# -- End of local dependencies install --",
        os.linesep.join(env_vars),
        "",
        f"WORKDIR {local_deps.working_dir}" if local_deps.working_dir else "",
    ]

    additional_contexts: dict[str, str] = {}
    for p in local_deps.additional_contexts:
        if p in local_deps.real_pkgs:
            name = local_deps.real_pkgs[p][1]
        elif p in local_deps.faux_pkgs:
            name = f"__outer_{p.name}"
        else:
            raise RuntimeError(f"Unknown additional context: {p}")
        additional_contexts[name] = str(p)

    return os.linesep.join(docker_file_contents), additional_contexts


def node_config_to_docker(
    config_path: pathlib.Path, config: Config, base_image: str
) -> tuple[str, dict[str, str]]:
    faux_path = f"/deps/{config_path.parent.name}"

    def test_file(file_name):
        full_path = config_path.parent / file_name
        try:
            return full_path.is_file()
        except OSError:
            return False

    # inspired by `package-manager-detector`
    def get_pkg_manager_name():
        try:
            with open(config_path.parent / "package.json") as f:
                pkg = json.load(f)

                if (pkg_manager_name := pkg.get("packageManager")) and isinstance(
                    pkg_manager_name, str
                ):
                    return pkg_manager_name.lstrip("^").split("@")[0]

                if (
                    dev_engine_name := (
                        (pkg.get("devEngines") or {}).get("packageManager") or {}
                    ).get("name")
                ) and isinstance(dev_engine_name, str):
                    return dev_engine_name

                return None
        except Exception:
            return None

    npm, yarn, pnpm, bun = [
        test_file("package-lock.json"),
        test_file("yarn.lock"),
        test_file("pnpm-lock.yaml"),
        test_file("bun.lockb"),
    ]

    if yarn:
        install_cmd = "yarn install --frozen-lockfile"
    elif pnpm:
        install_cmd = "pnpm i --frozen-lockfile"
    elif npm:
        install_cmd = "npm ci"
    elif bun:
        install_cmd = "bun i"
    else:
        pkg_manager_name = get_pkg_manager_name()

        if pkg_manager_name == "yarn":
            install_cmd = "yarn install"
        elif pkg_manager_name == "pnpm":
            install_cmd = "pnpm i"
        elif pkg_manager_name == "bun":
            install_cmd = "bun i"
        else:
            install_cmd = "npm i"
    store_config = config.get("store")
    env_additional_config = (
        ""
        if not store_config
        else f"""
ENV LANGGRAPH_STORE='{json.dumps(store_config)}'
"""
    )
    if (auth_config := config.get("auth")) is not None:
        env_additional_config += f"""
ENV LANGGRAPH_AUTH='{json.dumps(auth_config)}'
"""
    if (http_config := config.get("http")) is not None:
        env_additional_config += f"""
ENV LANGGRAPH_HTTP='{json.dumps(http_config)}'
"""

    return (
        f"""FROM {base_image}:{config['node_version']}

{os.linesep.join(config["dockerfile_lines"])}

ADD . {faux_path}

RUN cd {faux_path} && {install_cmd}
{env_additional_config}
ENV LANGSERVE_GRAPHS='{json.dumps(config["graphs"])}'
{f"ENV LANGGRAPH_UI='{json.dumps(config['ui'])}'" if config.get("ui") else ""}

WORKDIR {faux_path}

RUN (test ! -f /api/langgraph_api/js/build.mts && echo "Prebuild script not found, skipping") || tsx /api/langgraph_api/js/build.mts""",
        {},
    )


def config_to_docker(
    config_path: pathlib.Path, config: Config, base_image: str
) -> tuple[str, dict[str, str]]:
    if config.get("node_version"):
        return node_config_to_docker(config_path, config, base_image)

    return python_config_to_docker(config_path, config, base_image)


def config_to_compose(
    config_path: pathlib.Path,
    config: Config,
    base_image: str,
    watch: bool = False,
) -> str:
    env_vars = config["env"].items() if isinstance(config["env"], dict) else {}
    env_vars_str = "\n".join(f'            {k}: "{v}"' for k, v in env_vars)
    env_file_str = (
        f"env_file: {config['env']}" if isinstance(config["env"], str) else ""
    )
    if watch:
        dependencies = config.get("dependencies") or ["."]
        watch_paths = [config_path.name] + [
            dep for dep in dependencies if dep.startswith(".")
        ]
        watch_actions = "\n".join(
            f"""- path: {path}
  action: rebuild"""
            for path in watch_paths
        )
        watch_str = f"""
        develop:
            watch:
{textwrap.indent(watch_actions, "                ")}
"""
    else:
        watch_str = ""

    dockerfile, additional_contexts = config_to_docker(config_path, config, base_image)

    additional_contexts_str = "\n".join(
        f"                - {name}: {path}"
        for name, path in additional_contexts.items()
    )
    if additional_contexts_str:
        additional_contexts_str = f"""
            additional_contexts:
{additional_contexts_str}"""

    return f"""
{textwrap.indent(env_vars_str, "            ")}
        {env_file_str}
        pull_policy: build
        build:
            context: .{additional_contexts_str}
            dockerfile_inline: |
{textwrap.indent(dockerfile, "                ")}
        {watch_str}
"""
