import os
import pathlib
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10.
    import tomli as tomllib

import click

from langgraph_cli.schemas import Config


@dataclass(frozen=True)
class UvLockSourceEntry:
    name: str
    value: object
    declared_root: pathlib.Path
    pyproject_path: pathlib.Path


@dataclass(frozen=True)
class DiscoveredUvLockPackage:
    name: str
    normalized_name: str
    root: pathlib.Path
    pyproject_path: pathlib.Path
    raw_dependency_specs: object
    raw_uv_tool: object


@dataclass(frozen=True)
class UvLockPackage:
    name: str
    normalized_name: str
    root: pathlib.Path
    pyproject_path: pathlib.Path
    package_enabled: bool
    dependency_names: tuple[str, ...]
    workspace_dependencies: tuple[str, ...]


@dataclass(frozen=True)
class UvLockWorkspace:
    raw_root_source_entries: object
    packages_by_name: dict[str, DiscoveredUvLockPackage]
    packages_by_root: dict[pathlib.Path, DiscoveredUvLockPackage]


@dataclass(frozen=True)
class UvLockPlan:
    project_root: pathlib.Path
    pyproject_path: pathlib.Path
    uv_lock_path: pathlib.Path
    target: UvLockPackage
    install_order: tuple[UvLockPackage, ...]
    container_roots: dict[pathlib.Path, pathlib.PurePosixPath]
    working_dir: str


def _normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_dependency_name(
    dependency: str, *, package_name: str, pyproject_path: pathlib.Path
) -> str:
    match = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)", dependency)
    if not match:
        raise click.UsageError(
            "pip_installer 'uv_lock' only supports PEP 508 dependency strings "
            f"with an explicit package name. Could not parse dependency "
            f"{dependency!r} in {pyproject_path} for package '{package_name}'."
        )
    return _normalize_package_name(match.group(1))


def _load_pyproject(pyproject_path: pathlib.Path) -> dict:
    with pyproject_path.open("rb") as pyproject_file:
        return tomllib.load(pyproject_file)


def _get_dependency_names(
    dependency_specs: object, *, package_name: str, pyproject_path: pathlib.Path
) -> tuple[str, ...]:
    if dependency_specs is None:
        dependency_specs = []
    if not isinstance(dependency_specs, list) or not all(
        isinstance(spec, str) for spec in dependency_specs
    ):
        raise click.UsageError(
            "pip_installer 'uv_lock' requires [project].dependencies to be a "
            f"list of strings in {pyproject_path}."
        )
    return tuple(
        _parse_dependency_name(
            spec,
            package_name=package_name,
            pyproject_path=pyproject_path,
        )
        for spec in dependency_specs
    )


def _validate_uv_lock_source_entry(
    *,
    source_name: str,
    source_value: object,
    declared_root: pathlib.Path,
    pyproject_path: pathlib.Path,
    project_root: pathlib.Path,
    packages_by_name: dict[str, DiscoveredUvLockPackage],
    packages_by_root: dict[pathlib.Path, DiscoveredUvLockPackage],
) -> set[str]:
    workspace_dependencies: set[str] = set()
    if isinstance(source_value, list):
        for item in source_value:
            workspace_dependencies.update(
                _validate_uv_lock_source_entry(
                    source_name=source_name,
                    source_value=item,
                    declared_root=declared_root,
                    pyproject_path=pyproject_path,
                    project_root=project_root,
                    packages_by_name=packages_by_name,
                    packages_by_root=packages_by_root,
                )
            )
        return workspace_dependencies

    if not isinstance(source_value, dict):
        return workspace_dependencies

    normalized_source_name = _normalize_package_name(source_name)

    if source_value.get("workspace") is True:
        package = packages_by_name.get(normalized_source_name)
        if package is None:
            raise click.UsageError(
                "pip_installer 'uv_lock' only supports workspace-local [tool.uv.sources] "
                f"entries under project_root. '{source_name}' in {pyproject_path} is "
                "marked as a workspace dependency but does not resolve to a workspace "
                "package under project_root."
            )
        if not _get_uv_lock_package_enabled(package):
            raise click.UsageError(
                "pip_installer 'uv_lock' does not support workspace dependencies "
                f"with `tool.uv.package = false`. '{source_name}' in {pyproject_path} "
                "must be a buildable package."
            )
        workspace_dependencies.add(normalized_source_name)

    path_ref = source_value.get("path")
    if isinstance(path_ref, str):
        if (
            pathlib.Path(path_ref).is_absolute()
            or pathlib.PurePosixPath(path_ref).is_absolute()
        ):
            raise click.UsageError(
                "pip_installer 'uv_lock' does not support absolute [tool.uv.sources] "
                f"paths: {path_ref} in {pyproject_path}."
            )

        resolved = (declared_root / path_ref).resolve()
        if project_root != resolved and project_root not in resolved.parents:
            raise click.UsageError(
                "pip_installer 'uv_lock' only supports workspace dependencies "
                "declared with `{ workspace = true }` under project_root. "
                f"'{source_name}' in {pyproject_path} points to {resolved}, which "
                f"is outside project_root {project_root}."
            )

        if resolved not in packages_by_root:
            raise click.UsageError(
                "pip_installer 'uv_lock' only supports workspace dependencies "
                "declared with `{ workspace = true }` under project_root. "
                f"'{source_name}' in {pyproject_path} points to {resolved}, which "
                "is not a declared workspace package."
            )

        raise click.UsageError(
            "pip_installer 'uv_lock' only supports workspace dependencies "
            "declared with `{ workspace = true }`. "
            f"Replace the path source for '{source_name}' in {pyproject_path}."
        )

    for nested_value in source_value.values():
        workspace_dependencies.update(
            _validate_uv_lock_source_entry(
                source_name=source_name,
                source_value=nested_value,
                declared_root=declared_root,
                pyproject_path=pyproject_path,
                project_root=project_root,
                packages_by_name=packages_by_name,
                packages_by_root=packages_by_root,
            )
        )
    return workspace_dependencies


def _get_uv_lock_package_enabled(package: DiscoveredUvLockPackage) -> bool:
    uv_tool = package.raw_uv_tool
    if uv_tool and not isinstance(uv_tool, dict):
        raise click.UsageError(
            "pip_installer 'uv_lock' requires [tool.uv] to be a table "
            f"in {package.pyproject_path}."
        )
    package_enabled = (
        uv_tool.get("package", True) if isinstance(uv_tool, dict) else True
    )
    if not isinstance(package_enabled, bool):
        raise click.UsageError(
            "pip_installer 'uv_lock' requires [tool.uv].package to be a boolean "
            f"in {package.pyproject_path}."
        )
    return package_enabled


def _get_uv_lock_source_entries(
    package: DiscoveredUvLockPackage,
    *,
    project_root: pathlib.Path,
    root_pyproject_path: pathlib.Path,
    raw_root_source_entries: object,
) -> tuple[UvLockSourceEntry, ...]:
    if not isinstance(raw_root_source_entries, dict):
        raise click.UsageError(
            "pip_installer 'uv_lock' requires [tool.uv.sources] to be a table "
            f"in {root_pyproject_path}."
        )

    uv_tool = package.raw_uv_tool
    package_source_entries = (
        uv_tool.get("sources", {}) if isinstance(uv_tool, dict) else {}
    )
    if not isinstance(package_source_entries, dict):
        raise click.UsageError(
            "pip_installer 'uv_lock' requires [tool.uv.sources] to be a table "
            f"in {package.pyproject_path}."
        )

    source_entries: dict[str, UvLockSourceEntry] = {
        source_name: UvLockSourceEntry(
            name=source_name,
            value=source_value,
            declared_root=project_root,
            pyproject_path=root_pyproject_path,
        )
        for source_name, source_value in raw_root_source_entries.items()
    }
    source_entries.update(
        {
            source_name: UvLockSourceEntry(
                name=source_name,
                value=source_value,
                declared_root=package.root,
                pyproject_path=package.pyproject_path,
            )
            for source_name, source_value in package_source_entries.items()
        }
    )
    return tuple(source_entries.values())


def _validate_uv_lock_package(
    package: DiscoveredUvLockPackage,
    *,
    project_root: pathlib.Path,
    root_pyproject_path: pathlib.Path,
    raw_root_source_entries: object,
    packages_by_name: dict[str, DiscoveredUvLockPackage],
    packages_by_root: dict[pathlib.Path, DiscoveredUvLockPackage],
) -> UvLockPackage:
    dependency_names = _get_dependency_names(
        package.raw_dependency_specs,
        package_name=package.name,
        pyproject_path=package.pyproject_path,
    )
    dependency_name_set = set(dependency_names)

    workspace_dependency_names: set[str] = set()
    for source_entry in _get_uv_lock_source_entries(
        package,
        project_root=project_root,
        root_pyproject_path=root_pyproject_path,
        raw_root_source_entries=raw_root_source_entries,
    ):
        if _normalize_package_name(source_entry.name) not in dependency_name_set:
            continue
        workspace_dependency_names.update(
            _validate_uv_lock_source_entry(
                source_name=source_entry.name,
                source_value=source_entry.value,
                declared_root=source_entry.declared_root,
                pyproject_path=source_entry.pyproject_path,
                project_root=project_root,
                packages_by_name=packages_by_name,
                packages_by_root=packages_by_root,
            )
        )

    return UvLockPackage(
        name=package.name,
        normalized_name=package.normalized_name,
        root=package.root,
        pyproject_path=package.pyproject_path,
        package_enabled=_get_uv_lock_package_enabled(package),
        dependency_names=dependency_names,
        workspace_dependencies=tuple(
            dependency_name
            for dependency_name in dependency_names
            if dependency_name in workspace_dependency_names
        ),
    )


def _discover_uv_lock_workspace_packages(
    project_root: pathlib.Path, pyproject_path: pathlib.Path
) -> UvLockWorkspace:
    root_data = _load_pyproject(pyproject_path)
    root_source_entries = root_data.get("tool", {}).get("uv", {}).get("sources", {})

    candidate_roots: list[pathlib.Path] = []
    root_project = root_data.get("project", {})
    if isinstance(root_project, dict) and isinstance(root_project.get("name"), str):
        candidate_roots.append(project_root)

    workspace_members = (
        root_data.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])
    )
    if workspace_members and not isinstance(workspace_members, list):
        raise click.UsageError(
            "pip_installer 'uv_lock' requires [tool.uv.workspace].members to be a list."
        )

    for pattern in workspace_members:
        if not isinstance(pattern, str):
            raise click.UsageError(
                "pip_installer 'uv_lock' requires every [tool.uv.workspace].members "
                "entry to be a string."
            )
        for match in sorted(project_root.glob(pattern)):
            package_root = match if match.is_dir() else match.parent
            package_root = package_root.resolve()
            if (package_root / "pyproject.toml").is_file():
                candidate_roots.append(package_root)

    unique_roots: list[pathlib.Path] = []
    seen_roots: set[pathlib.Path] = set()
    for root in candidate_roots:
        if root not in seen_roots:
            unique_roots.append(root)
            seen_roots.add(root)

    packages: list[DiscoveredUvLockPackage] = []
    for package_root in unique_roots:
        member_pyproject_path = package_root / "pyproject.toml"
        pyproject_data = _load_pyproject(member_pyproject_path)

        project_data = pyproject_data.get("project", {})
        package_name = (
            project_data.get("name") if isinstance(project_data, dict) else None
        )
        if not isinstance(package_name, str) or not package_name.strip():
            raise click.UsageError(
                "pip_installer 'uv_lock' requires every workspace package to define "
                f"[project].name in {member_pyproject_path}."
            )

        packages.append(
            DiscoveredUvLockPackage(
                name=package_name,
                normalized_name=_normalize_package_name(package_name),
                root=package_root,
                pyproject_path=member_pyproject_path,
                raw_dependency_specs=project_data.get("dependencies", []),
                raw_uv_tool=pyproject_data.get("tool", {}).get("uv", {}),
            )
        )

    packages_by_name: dict[str, DiscoveredUvLockPackage] = {}
    packages_by_root: dict[pathlib.Path, DiscoveredUvLockPackage] = {}
    for package in packages:
        existing = packages_by_name.get(package.normalized_name)
        if existing is not None:
            raise click.UsageError(
                "pip_installer 'uv_lock' requires unique workspace package names, "
                f"but both {existing.pyproject_path} and {package.pyproject_path} "
                f"define '{package.name}'."
            )
        packages_by_name[package.normalized_name] = package
        packages_by_root[package.root] = package

    return UvLockWorkspace(
        raw_root_source_entries=root_source_entries,
        packages_by_name=packages_by_name,
        packages_by_root=packages_by_root,
    )


def _container_workspace_root() -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("/deps/workspace")


def _container_root_for_uv_lock_package(
    project_root: pathlib.Path, package_root: pathlib.Path
) -> pathlib.PurePosixPath:
    container_root = _container_workspace_root()
    relative_root = package_root.relative_to(project_root)
    if relative_root == pathlib.Path("."):
        return container_root
    return container_root.joinpath(*relative_root.parts)


def _resolve_uv_lock_container_path(
    host_path: pathlib.Path, plan: UvLockPlan
) -> pathlib.PurePosixPath | None:
    for package_root in sorted(
        plan.container_roots, key=lambda path: len(path.parts), reverse=True
    ):
        if host_path == package_root or package_root in host_path.parents:
            relative_path = host_path.relative_to(package_root)
            container_root = plan.container_roots[package_root]
            if relative_path == pathlib.Path("."):
                return container_root
            return container_root.joinpath(*relative_path.parts)
    return None


def _plan_uv_lock_workspace(config_path: pathlib.Path, config: Config) -> UvLockPlan:
    config_root = config_path.parent.resolve()
    project_root = (config_root / config["project_root"]).resolve()
    pyproject_path = project_root / "pyproject.toml"
    uv_lock_path = project_root / "uv.lock"

    if not uv_lock_path.exists():
        raise click.UsageError(
            f"pip_installer is 'uv_lock' but no uv.lock file found at {uv_lock_path}. "
            "Ensure project_root points to a workspace root containing uv.lock."
        )
    if not pyproject_path.exists():
        raise click.UsageError(
            f"pip_installer is 'uv_lock' but no pyproject.toml found at {pyproject_path}. "
            "Ensure project_root points to a workspace root containing pyproject.toml."
        )

    workspace = _discover_uv_lock_workspace_packages(project_root, pyproject_path)
    packages_by_name = workspace.packages_by_name
    target_name = _normalize_package_name(config["package"])
    target = packages_by_name.get(target_name)
    if target is None:
        available_packages = ", ".join(
            sorted(package.name for package in packages_by_name.values())
        )
        raise click.UsageError(
            f"pip_installer 'uv_lock' could not find package '{config['package']}' "
            f"under project_root {project_root}. Available workspace packages: "
            f"{available_packages or '(none)'}."
        )
    target_package = _validate_uv_lock_package(
        target,
        project_root=project_root,
        root_pyproject_path=pyproject_path,
        raw_root_source_entries=workspace.raw_root_source_entries,
        packages_by_name=workspace.packages_by_name,
        packages_by_root=workspace.packages_by_root,
    )
    if not target_package.package_enabled:
        raise click.UsageError(
            "pip_installer 'uv_lock' requires `package` to reference a buildable "
            f"workspace package. '{target_package.name}' sets `tool.uv.package = false`."
        )

    install_order: list[UvLockPackage] = []
    visited: set[str] = set()
    validated_packages: dict[str, UvLockPackage] = {
        target_package.normalized_name: target_package
    }

    def visit(package: DiscoveredUvLockPackage) -> None:
        if package.normalized_name in visited:
            return
        visited.add(package.normalized_name)
        validated_package = validated_packages.get(package.normalized_name)
        if validated_package is None:
            validated_package = _validate_uv_lock_package(
                package,
                project_root=project_root,
                root_pyproject_path=pyproject_path,
                raw_root_source_entries=workspace.raw_root_source_entries,
                packages_by_name=workspace.packages_by_name,
                packages_by_root=workspace.packages_by_root,
            )
            validated_packages[package.normalized_name] = validated_package
        for dependency_name in validated_package.workspace_dependencies:
            dependency = packages_by_name.get(dependency_name)
            if dependency is not None:
                visit(dependency)
        install_order.append(validated_package)

    visit(target)

    container_roots = {
        package.root: _container_root_for_uv_lock_package(project_root, package.root)
        for package in install_order
    }

    working_dir = _resolve_uv_lock_container_path(
        config_root,
        UvLockPlan(
            project_root=project_root,
            pyproject_path=pyproject_path,
            uv_lock_path=uv_lock_path,
            target=target_package,
            install_order=tuple(install_order),
            container_roots=container_roots,
            working_dir=str(
                _container_root_for_uv_lock_package(project_root, target_package.root)
            ),
        ),
    )
    if working_dir is None:
        working_dir = container_roots[target_package.root]

    return UvLockPlan(
        project_root=project_root,
        pyproject_path=pyproject_path,
        uv_lock_path=uv_lock_path,
        target=target_package,
        install_order=tuple(install_order),
        container_roots=container_roots,
        working_dir=str(working_dir),
    )


def _rewrite_uv_lock_import_path(
    config_path: pathlib.Path,
    import_str: str,
    plan: UvLockPlan,
    *,
    label: str,
) -> str:
    module_str, _, attr_str = import_str.partition(":")
    if not module_str or not attr_str:
        raise ValueError(
            f'Import string "{import_str}" must be in format "<module>:<attribute>".'
        )

    if "/" not in module_str and "\\" not in module_str:
        return import_str

    resolved = (config_path.parent / module_str).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Could not find {label}: {resolved}")
    if not resolved.is_file():
        raise IsADirectoryError(f"{label.capitalize()} must be a file: {resolved}")

    container_path = _resolve_uv_lock_container_path(resolved, plan)
    if container_path is None:
        raise click.UsageError(
            f"{label.capitalize()} '{import_str}' is not inside the target package "
            f"'{plan.target.name}' or its workspace package dependencies."
        )

    return f"{container_path.as_posix()}:{attr_str}"


def _update_uv_lock_graph_paths(
    config_path: pathlib.Path, config: Config, plan: UvLockPlan
) -> None:
    for graph_id, data in config["graphs"].items():
        if isinstance(data, dict):
            if "path" not in data:
                raise ValueError(
                    f"Graph '{graph_id}' must contain a 'path' key if it is a dictionary."
                )
            config["graphs"][graph_id]["path"] = _rewrite_uv_lock_import_path(
                config_path,
                data["path"],
                plan,
                label=f"graph '{graph_id}'",
            )
        elif isinstance(data, str):
            config["graphs"][graph_id] = _rewrite_uv_lock_import_path(
                config_path,
                data,
                plan,
                label=f"graph '{graph_id}'",
            )
        else:
            raise ValueError(
                f"Graph '{graph_id}' must be a string or a dictionary with a 'path' key."
            )


def _update_uv_lock_component_path(
    config_path: pathlib.Path,
    config: Config,
    plan: UvLockPlan,
    *,
    section: str,
    key: str,
    label: str,
) -> None:
    section_config = config.get(section)
    if not isinstance(section_config, dict):
        return

    path_str = section_config.get(key)
    if not isinstance(path_str, str):
        return

    section_config[key] = _rewrite_uv_lock_import_path(
        config_path,
        path_str,
        plan,
        label=label,
    )


def python_config_to_docker_uv_lock(
    config_path: pathlib.Path,
    config: Config,
    base_image: str,
    api_version: str | None = None,
    *,
    build_tools_to_uninstall: tuple[str] | None,
    image_supports_uv: Callable[[str], bool],
    get_node_pm_install_cmd: Callable[[pathlib.Path, Config], str],
    get_pip_cleanup_lines: Callable[[str, tuple[str] | None, str], str],
    docker_tag: Callable[[Config, str | None, str | None], str],
    build_python_install_commands: Callable[[Config, str], tuple[str, str, str]],
    build_runtime_env_vars: Callable[[Config], list[str]],
    default_node_version: str,
) -> tuple[str, dict[str, str]]:
    if not image_supports_uv(base_image):
        raise ValueError(
            "pip_installer 'uv_lock' requires a base image with uv support "
            "(langchain/langgraph-api >= 0.2.47)"
        )

    config_root = config_path.parent.resolve()
    install_cmd = "uv pip install --system"
    _, global_reqs_pip_install, pip_config_file_str = build_python_install_commands(
        config, install_cmd
    )
    plan = _plan_uv_lock_workspace(config_path, config)

    _update_uv_lock_graph_paths(config_path, config, plan)
    _update_uv_lock_component_path(
        config_path,
        config,
        plan,
        section="auth",
        key="path",
        label="auth.path",
    )
    _update_uv_lock_component_path(
        config_path,
        config,
        plan,
        section="encryption",
        key="path",
        label="encryption.path",
    )
    _update_uv_lock_component_path(
        config_path,
        config,
        plan,
        section="checkpointer",
        key="path",
        label="checkpointer.path",
    )
    _update_uv_lock_component_path(
        config_path,
        config,
        plan,
        section="http",
        key="app",
        label="http.app",
    )

    additional_contexts: dict[str, str] = {}
    workspace_context_name: str | None = None
    if (
        plan.project_root != config_root
        and config_root not in plan.project_root.parents
    ):
        workspace_context_name = "uv-workspace-root"
        additional_contexts[workspace_context_name] = str(plan.project_root)

    def copy_from_project_root(
        relative_path: pathlib.PurePosixPath, destination: str
    ) -> str:
        if workspace_context_name is not None:
            source = relative_path.as_posix() or "."
            return f"COPY --from={workspace_context_name} {source} {destination}"

        source_path = plan.project_root / pathlib.Path(relative_path)
        relative_source = source_path.relative_to(config_root).as_posix()
        return f"ADD {relative_source} {destination}"

    uv_export_project_dir = "/tmp/uv_export/project"
    uv_lock_str = f"""# -- Installing dependencies from uv.lock --
{copy_from_project_root(pathlib.PurePosixPath("pyproject.toml"), f"{uv_export_project_dir}/pyproject.toml")}
{copy_from_project_root(pathlib.PurePosixPath("uv.lock"), f"{uv_export_project_dir}/uv.lock")}
RUN cd {uv_export_project_dir} && uv export --package {shlex.quote(config["package"])} --frozen --no-hashes --no-emit-project --no-emit-workspace -o uv_requirements.txt
RUN cd {uv_export_project_dir} && {global_reqs_pip_install} -r uv_requirements.txt
RUN rm -rf /tmp/uv_export
# -- End of uv.lock dependencies install --"""

    local_pkgs_str = os.linesep.join(
        [
            f"""# -- Adding workspace package {package.root.relative_to(plan.project_root).as_posix() or "."} --
{copy_from_project_root(pathlib.PurePosixPath(*package.root.relative_to(plan.project_root).parts), plan.container_roots[package.root].as_posix())}
# -- End of workspace package {package.root.relative_to(plan.project_root).as_posix() or "."} --"""
            for package in plan.install_order
        ]
    )

    install_local_pkgs_str = os.linesep.join(
        [
            (
                f"RUN cd {plan.container_roots[package.root].as_posix()} && "
                f"{global_reqs_pip_install} --no-deps -e ."
            )
            for package in plan.install_order
        ]
    )

    install_node_str = (
        "RUN /storage/install-node.sh"
        if (config.get("ui") or config.get("node_version")) and plan.working_dir
        else ""
    )
    installs = f"{os.linesep}{os.linesep}".join(
        filter(
            None, [install_node_str, pip_config_file_str, uv_lock_str, local_pkgs_str]
        )
    )

    env_vars = build_runtime_env_vars(config)

    js_inst_str = ""
    if (config.get("ui") or config.get("node_version")) and plan.working_dir:
        js_inst_str = os.linesep.join(
            [
                "# -- Installing JS dependencies --",
                f"ENV NODE_VERSION={config.get('node_version') or default_node_version}",
                f"RUN cd {plan.working_dir} && {get_node_pm_install_cmd(config_path, config)} && tsx /api/langgraph_api/js/build.mts",
                "# -- End of JS dependencies install --",
            ]
        )

    image_str = docker_tag(config, base_image, api_version)
    docker_file_contents = []

    if additional_contexts:
        docker_file_contents.extend(
            [
                "# syntax=docker/dockerfile:1.4",
                "",
            ]
        )

    docker_file_contents.extend(
        [
            f"FROM {image_str}",
            "",
            os.linesep.join(config["dockerfile_lines"]),
            "",
            installs,
            "",
            "# -- Installing workspace packages --",
            install_local_pkgs_str,
            "# -- End of workspace packages install --",
            os.linesep.join(env_vars),
            "",
            js_inst_str,
            "",
            get_pip_cleanup_lines(
                install_cmd=install_cmd,
                to_uninstall=build_tools_to_uninstall,
                pip_installer="uv",
            ),
            "",
            f"WORKDIR {plan.working_dir}" if plan.working_dir else "",
        ]
    )

    return os.linesep.join(docker_file_contents), additional_contexts
