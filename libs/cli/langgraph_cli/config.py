import json
import os
import pathlib
import textwrap
from typing import NamedTuple, Optional, TypedDict, Union

import click

MIN_NODE_VERSION = "20"
MIN_PYTHON_VERSION = "3.11"


class IndexConfig(TypedDict, total=False):
    """Configuration for indexing documents for semantic search in the store."""

    dims: int
    """Number of dimensions in the embedding vectors.
    
    Common embedding models have the following dimensions:
        - openai:text-embedding-3-large: 3072
        - openai:text-embedding-3-small: 1536
        - openai:text-embedding-ada-002: 1536
        - cohere:embed-english-v3.0: 1024
        - cohere:embed-english-light-v3.0: 384
        - cohere:embed-multilingual-v3.0: 1024
        - cohere:embed-multilingual-light-v3.0: 384
    """

    embed: str
    """Optional model (string) to generate embeddings from text or path to model or function.
    
    Examples:
        - "openai:text-embedding-3-large"
        - "cohere:embed-multilingual-v3.0"
        - "src/app.py:embeddings
    """

    fields: Optional[list[str]]
    """Fields to extract text from for embedding generation.
    
    Defaults to the root ["$"], which embeds the json object as a whole.
    """


class StoreConfig(TypedDict, total=False):
    embed: Optional[IndexConfig]
    """Configuration for vector embeddings in store."""


class SecurityConfig(TypedDict, total=False):
    securitySchemes: dict
    security: list
    # path => {method => security}
    paths: dict[str, dict[str, list]]


class AuthConfig(TypedDict, total=False):
    path: str
    """Path to the authentication function in a Python file."""
    disable_studio_auth: bool
    """Whether to disable auth when connecting from the LangSmith Studio."""
    openapi: SecurityConfig
    """The schema to use for updating the openapi spec.

    Example:
        {
            "securitySchemes": {
                "OAuth2": {
                    "type": "oauth2",
                    "flows": {
                        "password": {
                            "tokenUrl": "/token",
                            "scopes": {
                                "me": "Read information about the current user",
                                "items": "Access to create and manage items"
                            }
                        }
                    }
                }
            },
            "security": [
                {"OAuth2": ["me"]}  # Default security requirement for all endpoints
            ]
        }
    """


class Config(TypedDict, total=False):
    python_version: str
    node_version: Optional[str]
    pip_config_file: Optional[str]
    dockerfile_lines: list[str]
    dependencies: list[str]
    graphs: dict[str, str]
    env: Union[dict[str, str], str]
    store: Optional[StoreConfig]
    auth: Optional[AuthConfig]


def _parse_version(version_str: str) -> tuple[int, int]:
    """Parse a version string into a tuple of (major, minor)."""
    try:
        major, minor = map(int, version_str.split("."))
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
    config = (
        {
            "node_version": config.get("node_version"),
            "dockerfile_lines": config.get("dockerfile_lines", []),
            "graphs": config.get("graphs", {}),
            "env": config.get("env", {}),
            "store": config.get("store"),
            "auth": config.get("auth"),
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
            part.isdigit() for part in pyversion.split(".")
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
    return config


def validate_config_file(config_path: pathlib.Path) -> Config:
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
    pip_reqs: list[tuple[pathlib.Path, str]]
    real_pkgs: dict[pathlib.Path, str]
    faux_pkgs: dict[pathlib.Path, tuple[str, str]]
    # if . is in dependencies, use it as working_dir
    working_dir: Optional[str] = None


def _assemble_local_deps(config_path: pathlib.Path, config: Config) -> LocalDeps:
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
    working_dir = None

    for local_dep in config["dependencies"]:
        if not local_dep.startswith("."):
            continue

        resolved = config_path.parent / local_dep

        # validate local dependency
        if not resolved.exists():
            raise FileNotFoundError(f"Could not find local dependency: {resolved}")
        elif not resolved.is_dir():
            raise NotADirectoryError(
                f"Local dependency must be a directory: {resolved}"
            )
        elif not resolved.is_relative_to(config_path.parent):
            raise ValueError(
                f"Local dependency '{resolved}' must be a subdirectory of '{config_path.parent}'"
            )

        # if it's installable, add it to local_pkgs
        # otherwise, add it to faux_pkgs, and create a pyproject.toml
        files = os.listdir(resolved)
        if "pyproject.toml" in files:
            real_pkgs[resolved] = local_dep
            if local_dep == ".":
                working_dir = f"/deps/{resolved.name}"
        elif "setup.py" in files:
            real_pkgs[resolved] = local_dep
            if local_dep == ".":
                working_dir = f"/deps/{resolved.name}"
        else:
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
            if "requirements.txt" in files:
                rfile = resolved / "requirements.txt"
                pip_reqs.append(
                    (
                        rfile.relative_to(config_path.parent),
                        f"{container_path}/requirements.txt",
                    )
                )

    return LocalDeps(pip_reqs, real_pkgs, faux_pkgs, working_dir)


def _update_graph_paths(
    config_path: pathlib.Path, config: Config, local_deps: LocalDeps
) -> None:
    for graph_id, import_str in config["graphs"].items():
        module_str, _, attr_str = import_str.partition(":")
        if not module_str or not attr_str:
            message = (
                'Import string "{import_str}" must be in format "<module>:<attribute>".'
            )
            raise ValueError(message.format(import_str=import_str))
        if "/" in module_str:
            resolved = config_path.parent / module_str
            if not resolved.exists():
                raise FileNotFoundError(f"Could not find local module: {resolved}")
            elif not resolved.is_file():
                raise IsADirectoryError(f"Local module must be a file: {resolved}")
            else:
                for path in local_deps.real_pkgs:
                    if resolved.is_relative_to(path):
                        module_str = f"/deps/{path.name}/{resolved.relative_to(path)}"
                        break
                else:
                    for faux_pkg, (_, destpath) in local_deps.faux_pkgs.items():
                        if resolved.is_relative_to(faux_pkg):
                            module_str = f"{destpath}/{resolved.relative_to(faux_pkg)}"
                            break
                    else:
                        raise ValueError(
                            f"Module '{import_str}' not found in 'dependencies' list. "
                            "Add its containing package to 'dependencies' list."
                        )
            # update the config
            config["graphs"][graph_id] = f"{module_str}:{attr_str}"


def python_config_to_docker(config_path: pathlib.Path, config: Config, base_image: str):
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

    # rewrite graph paths
    _update_graph_paths(config_path, config, local_deps)

    pip_pkgs_str = f"RUN {pip_install} {' '.join(pypi_deps)}" if pypi_deps else ""
    if local_deps.pip_reqs:
        pip_reqs_str = os.linesep.join(
            f"ADD {reqpath} {destpath}" for reqpath, destpath in local_deps.pip_reqs
        )
        pip_reqs_str += f'{os.linesep}RUN {pip_install} {" ".join("-r " + r for _,r in local_deps.pip_reqs)}'

    else:
        pip_reqs_str = ""

    # https://setuptools.pypa.io/en/latest/userguide/datafiles.html#package-data
    # https://til.simonwillison.net/python/pyproject
    faux_pkgs_str = f"{os.linesep}{os.linesep}".join(
        f"""ADD {relpath} {destpath}
RUN set -ex && \\
    for line in '[project]' \\
                'name = "{fullpath.name}"' \\
                'version = "0.1"' \\
                '[tool.setuptools.package-data]' \\
                '"*" = ["**/*"]'; do \\
        echo "$line" >> /deps/__outer_{fullpath.name}/pyproject.toml; \\
    done"""
        for fullpath, (relpath, destpath) in local_deps.faux_pkgs.items()
    )
    local_pkgs_str = os.linesep.join(
        f"ADD {relpath} /deps/{fullpath.name}"
        for fullpath, relpath in local_deps.real_pkgs.items()
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
    return f"""FROM {base_image}:{config['python_version']}

{os.linesep.join(config["dockerfile_lines"])}

{installs}

RUN {pip_install} -e /deps/*
{env_additional_config}
ENV LANGSERVE_GRAPHS='{json.dumps(config["graphs"])}'

{f"WORKDIR {local_deps.working_dir}" if local_deps.working_dir else ""}"""


def node_config_to_docker(config_path: pathlib.Path, config: Config, base_image: str):
    faux_path = f"/deps/{config_path.parent.name}"

    def test_file(file_name):
        full_path = config_path.parent / file_name
        try:
            return full_path.is_file()
        except OSError:
            return False

    npm, yarn, pnpm = [
        test_file("package-lock.json"),
        test_file("yarn.lock"),
        test_file("pnpm-lock.yaml"),
    ]

    if yarn:
        install_cmd = "yarn install --frozen-lockfile"
    elif pnpm:
        install_cmd = "pnpm i --frozen-lockfile"
    elif npm:
        install_cmd = "npm ci"
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
    return f"""FROM {base_image}:{config['node_version']}

{os.linesep.join(config["dockerfile_lines"])}

ADD . {faux_path}

RUN cd {faux_path} && {install_cmd}
{env_additional_config}
ENV LANGSERVE_GRAPHS='{json.dumps(config["graphs"])}'

WORKDIR {faux_path}

RUN (test ! -f /api/langgraph_api/js/build.mts && echo "Prebuild script not found, skipping") || tsx /api/langgraph_api/js/build.mts"""


def config_to_docker(config_path: pathlib.Path, config: Config, base_image: str):
    if config.get("node_version"):
        return node_config_to_docker(config_path, config, base_image)

    return python_config_to_docker(config_path, config, base_image)


def config_to_compose(
    config_path: pathlib.Path,
    config: Config,
    base_image: str,
    watch: bool = False,
):
    env_vars = config["env"].items() if isinstance(config["env"], dict) else {}
    env_vars_str = "\n".join(f'            {k}: "{v}"' for k, v in env_vars)
    env_file_str = (
        f"env_file: {config['env']}" if isinstance(config["env"], str) else ""
    )
    if watch:
        watch_paths = [config_path.name] + [
            dep for dep in config["dependencies"] if dep.startswith(".")
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

    return f"""
{textwrap.indent(env_vars_str, "            ")}
        {env_file_str}
        pull_policy: build
        build:
            context: .
            dockerfile_inline: |
{textwrap.indent(config_to_docker(config_path, config, base_image), "                ")}
        {watch_str}
"""
