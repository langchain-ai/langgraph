import json
import os
import pathlib
import textwrap
from typing import NamedTuple, Optional, TypedDict, Union

import click


class Config(TypedDict):
    python_version: str
    pip_config_file: Optional[str]
    dockerfile_lines: list[str]
    dependencies: list[str]
    graphs: dict[str, str]
    env: Union[dict[str, str], str]


def validate_config(config: Config) -> Config:
    config = {
        "python_version": config.get("python_version", "3.11"),
        "pip_config_file": config.get("pip_config_file"),
        "dockerfile_lines": config.get("dockerfile_lines", []),
        "dependencies": config.get("dependencies", []),
        "graphs": config.get("graphs", {}),
        "env": config.get("env", {}),
    }
    if config["python_version"] not in (
        "3.11",
        "3.12",
    ):
        raise click.UsageError(
            f"Unsupported Python version: {config['python_version']}. "
            "Supported versions are 3.11 and 3.12."
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


def config_to_docker(config_path: pathlib.Path, config: Config, base_image: str):
    # configure pip
    pip_install = "pip install -c /api/constraints.txt"
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
    done
"""
        for fullpath, (relpath, destpath) in local_deps.faux_pkgs.items()
    )
    local_pkgs_str = os.linesep.join(
        f"ADD {relpath} /deps/{fullpath.name}"
        for fullpath, relpath in local_deps.real_pkgs.items()
    )

    return f"""FROM {base_image}:{config['python_version']}

{os.linesep.join(config["dockerfile_lines"])}

{pip_config_file_str}

{pip_pkgs_str}

{pip_reqs_str}

{local_pkgs_str}

{faux_pkgs_str}

RUN {pip_install} -e /deps/*

ENV LANGSERVE_GRAPHS='{json.dumps(config["graphs"])}'

{f"WORKDIR {local_deps.working_dir}" if local_deps.working_dir else ""}"""


def config_to_compose(
    config_path: pathlib.Path,
    config: Config,
    base_image: str,
    watch: bool = False,
    langgraph_api_path: Optional[pathlib.Path] = None,
):
    env_vars = config["env"].items() if isinstance(config["env"], dict) else {}
    env_vars_str = "\n".join(f"            {k}: {v}" for k, v in env_vars)
    env_file_str = (
        f"env_file: {config['env']}" if isinstance(config["env"], str) else ""
    )
    if watch:
        watch_paths = [config_path] + [
            config_path.parent / dep
            for dep in config["dependencies"]
            if dep.startswith(".")
        ]
        watch_actions = "\n".join(
            f"""- path: {path}
  action: rebuild
  ignore:
    - .langgraph-data"""
            for path in watch_paths
        )
        if langgraph_api_path:
            watch_actions += f"""\n- path: {langgraph_api_path}
  action: sync+restart
  target: /api/langgraph_api"""
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
