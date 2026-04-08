import os
import pathlib
import re
import shlex
from dataclasses import dataclass

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10.
    import tomli as tomllib

import click

from langgraph_cli.schemas import Config


@dataclass(frozen=True, slots=True)
class UvLockSourceEntry:
    name: str
    value: object
    declared_root: pathlib.Path
    pyproject_path: pathlib.Path


@dataclass(slots=True)
class UvLockPackage:
    name: str
    normalized_name: str
    root: pathlib.Path
    pyproject_path: pathlib.Path
    raw_dependency_specs: object
    raw_uv_tool: object
    # Set after validation:
    package_enabled: bool = False
    dependency_names: tuple[str, ...] = ()
    workspace_dependencies: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class UvLockWorkspace:
    raw_root_source_entries: object
    packages_by_name: dict[str, UvLockPackage]
    packages_by_root: dict[pathlib.Path, UvLockPackage]


@dataclass(frozen=True, slots=True)
class UvLockPlan:
    project_root: pathlib.Path
    pyproject_path: pathlib.Path
    uv_lock_path: pathlib.Path
    target: UvLockPackage
    target_root: pathlib.Path
    install_order: tuple[UvLockPackage, ...]
    container_roots: dict[pathlib.Path, pathlib.PurePosixPath]
    working_dir: str
    all_workspace_roots: frozenset[pathlib.Path] = frozenset()


@dataclass(slots=True)
class DockerBuildPlan:
    lines: list[str]

    def add_blank(self) -> None:
        self.lines.append("")

    def add_raw(self, line: str) -> None:
        self.lines.append(line)

    def add_instruction(self, opcode: str, value: str | None = None) -> None:
        if value:
            self.lines.append(f"{opcode} {value}")
        else:
            self.lines.append(opcode)

    def extend_nonempty(self, items: list[str]) -> None:
        self.lines.extend(item for item in items if item)

    def render(self) -> str:
        return os.linesep.join(self.lines)


def _normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_dependency_name(
    dependency: str, *, package_name: str, pyproject_path: pathlib.Path
) -> str:
    match = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)", dependency)
    if not match:
        raise click.UsageError(
            "source.kind 'uv' only supports PEP 508 dependency strings "
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
            "source.kind 'uv' requires [project].dependencies to be a "
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
    packages_by_name: dict[str, UvLockPackage],
    packages_by_root: dict[pathlib.Path, UvLockPackage],
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
    package: UvLockPackage | None = None

    if source_value.get("workspace") is True:
        package = packages_by_name.get(normalized_source_name)
        if package is None:
            raise click.UsageError(
                f"'{source_name}' in {pyproject_path} is marked as "
                "`{{ workspace = true }}` but no matching workspace package was "
                f"found under project_root ({project_root}). Check that "
                f"'{source_name}' appears in [tool.uv.workspace].members."
            )
        if not _get_uv_lock_package_enabled(package):
            raise click.UsageError(
                f"'{source_name}' in {pyproject_path} is a workspace dependency "
                f"but sets `tool.uv.package = false` in {package.pyproject_path}. "
                "Workspace dependencies must be buildable packages."
            )
        workspace_dependencies.add(package.normalized_name)

    path_ref = source_value.get("path")
    if isinstance(path_ref, str):
        if (
            pathlib.Path(path_ref).is_absolute()
            or pathlib.PurePosixPath(path_ref).is_absolute()
        ):
            raise click.UsageError(
                f"'{source_name}' in {pyproject_path} uses an absolute path "
                f"({path_ref}), which is not supported. Use a relative path or "
                "`{{ workspace = true }}` instead."
            )

        resolved = (declared_root / path_ref).resolve()
        if project_root != resolved and project_root not in resolved.parents:
            raise click.UsageError(
                f"'{source_name}' in {pyproject_path} uses a path source that "
                f"resolves to {resolved}, which is outside project_root "
                f"({project_root})."
            )

        package = packages_by_root.get(resolved)
        if package is None:
            raise click.UsageError(
                f"'{source_name}' in {pyproject_path} uses a path source that "
                f"resolves to {resolved}, which is not a workspace package under "
                f"project_root ({project_root})."
            )
        if package.normalized_name != normalized_source_name:
            raise click.UsageError(
                f"'{source_name}' in {pyproject_path} points to {resolved}, which "
                f"defines package '{package.name}'. The dependency name and the "
                "workspace package name must match."
            )
        if not _get_uv_lock_package_enabled(package):
            raise click.UsageError(
                f"'{source_name}' in {pyproject_path} resolves to workspace package "
                f"'{package.name}', which sets `tool.uv.package = false` in "
                f"{package.pyproject_path}. Workspace dependencies must be "
                "buildable packages."
            )
        workspace_dependencies.add(package.normalized_name)

    for key, nested_value in source_value.items():
        if key in {"workspace", "path"}:
            continue
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


def _get_uv_lock_package_enabled(package: UvLockPackage) -> bool:
    uv_tool = package.raw_uv_tool
    if uv_tool and not isinstance(uv_tool, dict):
        raise click.UsageError(
            "source.kind 'uv' requires [tool.uv] to be a table "
            f"in {package.pyproject_path}."
        )
    package_enabled = (
        uv_tool.get("package", True) if isinstance(uv_tool, dict) else True
    )
    if not isinstance(package_enabled, bool):
        raise click.UsageError(
            "source.kind 'uv' requires [tool.uv].package to be a boolean "
            f"in {package.pyproject_path}."
        )
    return package_enabled


def _get_uv_lock_source_entries(
    package: UvLockPackage,
    *,
    project_root: pathlib.Path,
    root_pyproject_path: pathlib.Path,
    raw_root_source_entries: object,
) -> tuple[UvLockSourceEntry, ...]:
    if not isinstance(raw_root_source_entries, dict):
        raise click.UsageError(
            "source.kind 'uv' requires [tool.uv.sources] to be a table "
            f"in {root_pyproject_path}."
        )

    uv_tool = package.raw_uv_tool
    package_source_entries = (
        uv_tool.get("sources", {}) if isinstance(uv_tool, dict) else {}
    )
    if not isinstance(package_source_entries, dict):
        raise click.UsageError(
            "source.kind 'uv' requires [tool.uv.sources] to be a table "
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
    package: UvLockPackage,
    *,
    project_root: pathlib.Path,
    root_pyproject_path: pathlib.Path,
    raw_root_source_entries: object,
    packages_by_name: dict[str, UvLockPackage],
    packages_by_root: dict[pathlib.Path, UvLockPackage],
) -> None:
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

    package.package_enabled = _get_uv_lock_package_enabled(package)
    package.dependency_names = dependency_names
    package.workspace_dependencies = tuple(
        dependency_name
        for dependency_name in dependency_names
        if dependency_name in workspace_dependency_names
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
            "source.kind 'uv' requires [tool.uv.workspace].members to be a list."
        )

    for pattern in workspace_members:
        if not isinstance(pattern, str):
            raise click.UsageError(
                "source.kind 'uv' requires every [tool.uv.workspace].members "
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

    packages: list[UvLockPackage] = []
    for package_root in unique_roots:
        member_pyproject_path = package_root / "pyproject.toml"
        pyproject_data = _load_pyproject(member_pyproject_path)

        project_data = pyproject_data.get("project", {})
        package_name = (
            project_data.get("name") if isinstance(project_data, dict) else None
        )
        if not isinstance(package_name, str) or not package_name.strip():
            raise click.UsageError(
                "source.kind 'uv' requires every workspace package to define "
                f"[project].name in {member_pyproject_path}."
            )

        packages.append(
            UvLockPackage(
                name=package_name,
                normalized_name=_normalize_package_name(package_name),
                root=package_root,
                pyproject_path=member_pyproject_path,
                raw_dependency_specs=project_data.get("dependencies", []),
                raw_uv_tool=pyproject_data.get("tool", {}).get("uv", {}),
            )
        )

    packages_by_name: dict[str, UvLockPackage] = {}
    packages_by_root: dict[pathlib.Path, UvLockPackage] = {}
    for package in packages:
        existing = packages_by_name.get(package.normalized_name)
        if existing is not None:
            raise click.UsageError(
                "source.kind 'uv' requires unique workspace package names, "
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


def _uv_lock_package_copy_items(
    package: UvLockPackage, plan: UvLockPlan
) -> tuple[tuple[pathlib.PurePosixPath, pathlib.PurePosixPath], ...]:
    if package.root != plan.project_root:
        relative_root = pathlib.PurePosixPath(
            *package.root.relative_to(plan.project_root).parts
        )
        return ((relative_root, plan.container_roots[package.root]),)

    root_container = plan.container_roots[package.root]
    workspace_member_roots = plan.all_workspace_roots - {plan.project_root}

    def iter_entries(
        current_dir: pathlib.Path,
    ) -> tuple[tuple[pathlib.PurePosixPath, pathlib.PurePosixPath], ...]:
        entries: list[tuple[pathlib.PurePosixPath, pathlib.PurePosixPath]] = []
        for child in sorted(current_dir.iterdir(), key=lambda path: path.name):
            if child in workspace_member_roots:
                # Workspace members are copied separately if they are in the closure,
                # and excluded entirely otherwise.
                continue

            descendant_member_roots = [
                ws_root
                for ws_root in workspace_member_roots
                if child in ws_root.parents
            ]
            if child.is_dir() and descendant_member_roots:
                entries.extend(iter_entries(child))
                continue

            relative_child = pathlib.PurePosixPath(
                *child.relative_to(plan.project_root).parts
            )
            entries.append(
                (relative_child, root_container.joinpath(*relative_child.parts))
            )
        return tuple(entries)

    return iter_entries(plan.project_root)


def _resolve_uv_lock_container_path(
    host_path: pathlib.Path, plan: UvLockPlan
) -> pathlib.PurePosixPath | None:
    for package_root in sorted(
        plan.container_roots, key=lambda path: len(path.parts), reverse=True
    ):
        if host_path == package_root or package_root in host_path.parents:
            # Guard against the workspace root matching paths inside
            # unrelated workspace members that are not in the closure.
            if plan.all_workspace_roots and _path_in_unrelated_member(
                host_path, package_root, plan
            ):
                continue
            relative_path = host_path.relative_to(package_root)
            container_root = plan.container_roots[package_root]
            if relative_path == pathlib.Path("."):
                return container_root
            return container_root.joinpath(*relative_path.parts)
    return None


def _path_in_unrelated_member(
    host_path: pathlib.Path,
    matched_root: pathlib.Path,
    plan: UvLockPlan,
) -> bool:
    """Return True if host_path is inside a workspace member NOT in the closure.

    When matched_root is the project root (workspace root), it is a parent of
    every file in the workspace.  We need to reject paths that actually belong
    to a sibling workspace member that was not selected for deployment.
    """
    for ws_root in plan.all_workspace_roots:
        if ws_root == matched_root:
            continue
        # ws_root is more specific than matched_root
        if ws_root in matched_root.parents:
            continue
        if host_path == ws_root or ws_root in host_path.parents:
            # host_path is inside this workspace member
            if ws_root not in plan.container_roots:
                return True
    return False


def _infer_uv_lock_target_package(
    *,
    config_root: pathlib.Path,
    project_root: pathlib.Path,
    source: dict,
    packages_by_name: dict[str, UvLockPackage],
    packages_by_root: dict[pathlib.Path, UvLockPackage],
) -> UvLockPackage:
    package_name = source.get("package")
    if package_name is not None:
        if not isinstance(package_name, str) or not package_name.strip():
            raise click.UsageError("`source.package` must be a non-empty string.")
        target = packages_by_name.get(_normalize_package_name(package_name))
        if target is None:
            available_packages = ", ".join(
                sorted(package.name for package in packages_by_name.values())
            )
            raise click.UsageError(
                "Could not find source.package "
                f"'{package_name}' in the uv project at {project_root}. "
                "It must match a [project].name from one of the discovered "
                f"packages. Available packages: {available_packages or '(none)'}."
            )
        return target

    containing_packages = sorted(
        (
            package
            for package_root, package in packages_by_root.items()
            if config_root == package_root or package_root in config_root.parents
        ),
        key=lambda package: len(package.root.parts),
        reverse=True,
    )
    if containing_packages:
        target = containing_packages[0]
        if (
            target.root != project_root
            or len(packages_by_name) == 1
            or config_root == project_root
        ):
            return target

    if len(packages_by_name) == 1:
        return next(iter(packages_by_name.values()))

    available_packages = ", ".join(
        sorted(package.name for package in packages_by_name.values())
    )
    raise click.UsageError(
        "source.package is required because source.root resolves to a uv "
        "workspace with multiple packages and no unique target package could be "
        f"inferred from langgraph.json at {config_root}. Available packages: "
        f"{available_packages or '(none)'}. "
        "Move langgraph.json into the target package or set source.package."
    )


def _plan_uv_lock_workspace(config_path: pathlib.Path, config: Config) -> UvLockPlan:
    config_root = config_path.parent.resolve()
    source = config["source"]
    root = source.get("root", ".")
    if not isinstance(root, str) or not root.strip():
        raise click.UsageError('`source.root` must be a non-empty string. Use `"."`.')
    project_root = (config_root / root).resolve()
    pyproject_path = project_root / "pyproject.toml"
    uv_lock_path = project_root / "uv.lock"

    if not uv_lock_path.exists():
        raise click.UsageError(
            f"No uv.lock found at {uv_lock_path}. Your langgraph.json sets "
            f"source.root={root!r}, which resolves to "
            f"{project_root}. Make sure this is the directory where you run "
            "`uv lock` (it should contain both pyproject.toml and uv.lock)."
        )
    if not pyproject_path.exists():
        raise click.UsageError(
            f"No pyproject.toml found at {pyproject_path}. Your langgraph.json "
            f"sets source.root={root!r}, which resolves to "
            f"{project_root}. This should be your uv workspace root."
        )

    workspace = _discover_uv_lock_workspace_packages(project_root, pyproject_path)
    packages_by_name = workspace.packages_by_name
    target = _infer_uv_lock_target_package(
        config_root=config_root,
        project_root=project_root,
        source=source,
        packages_by_name=packages_by_name,
        packages_by_root=workspace.packages_by_root,
    )
    _validate_uv_lock_package(
        target,
        project_root=project_root,
        root_pyproject_path=pyproject_path,
        raw_root_source_entries=workspace.raw_root_source_entries,
        packages_by_name=workspace.packages_by_name,
        packages_by_root=workspace.packages_by_root,
    )
    if not target.package_enabled:
        raise click.UsageError(
            f"'{target.name}' has `tool.uv.package = false` in "
            f"{target.pyproject_path}, so it cannot be deployed. Either remove "
            "that setting or point `source.package` at a different workspace "
            "member."
        )

    install_order: list[UvLockPackage] = []
    visited: set[str] = set()
    validated: set[str] = {target.normalized_name}

    def visit(package: UvLockPackage) -> None:
        if package.normalized_name in visited:
            return
        visited.add(package.normalized_name)
        if package.normalized_name not in validated:
            _validate_uv_lock_package(
                package,
                project_root=project_root,
                root_pyproject_path=pyproject_path,
                raw_root_source_entries=workspace.raw_root_source_entries,
                packages_by_name=workspace.packages_by_name,
                packages_by_root=workspace.packages_by_root,
            )
            validated.add(package.normalized_name)
        for dependency_name in package.workspace_dependencies:
            dependency = packages_by_name.get(dependency_name)
            if dependency is not None:
                visit(dependency)
        install_order.append(package)

    visit(target)

    container_roots = {
        package.root: _container_root_for_uv_lock_package(project_root, package.root)
        for package in install_order
    }

    all_workspace_roots = frozenset(
        package.root for package in packages_by_name.values()
    )

    working_dir = _resolve_uv_lock_container_path(
        config_root,
        UvLockPlan(
            project_root=project_root,
            pyproject_path=pyproject_path,
            uv_lock_path=uv_lock_path,
            target=target,
            target_root=target.root,
            install_order=tuple(install_order),
            container_roots=container_roots,
            working_dir=str(
                _container_root_for_uv_lock_package(project_root, target.root)
            ),
            all_workspace_roots=all_workspace_roots,
        ),
    )
    if working_dir is None:
        working_dir = container_roots[target.root]

    return UvLockPlan(
        project_root=project_root,
        pyproject_path=pyproject_path,
        uv_lock_path=uv_lock_path,
        target=target,
        target_root=target.root,
        install_order=tuple(install_order),
        container_roots=container_roots,
        working_dir=str(working_dir),
        all_workspace_roots=all_workspace_roots,
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
        copied_dirs = ", ".join(
            package.root.relative_to(plan.project_root).as_posix() or "."
            for package in plan.install_order
        )
        raise click.UsageError(
            f"{label.capitalize()} '{import_str}' resolves to {resolved}, which is "
            f"not inside the target package '{plan.target.name}' or any of its "
            f"workspace dependencies. Only these directories are copied into the "
            f"container: {copied_dirs}. If this file lives in another workspace "
            f"package, add it as a dependency of '{plan.target.name}' with "
            "`{ workspace = true }` in [tool.uv.sources]."
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


def _update_uv_lock_ui_paths(
    config_path: pathlib.Path, config: Config, plan: UvLockPlan
) -> None:
    ui = config.get("ui")
    if not isinstance(ui, dict):
        return

    for ui_name, path_str in ui.items():
        if not isinstance(path_str, str):
            continue

        resolved = (config_path.parent / path_str).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Could not find ui '{ui_name}': {resolved}")
        if not resolved.is_file():
            raise IsADirectoryError(f"Ui '{ui_name}' must be a file: {resolved}")

        container_path = _resolve_uv_lock_container_path(resolved, plan)
        if container_path is None:
            copied_dirs = ", ".join(
                package.root.relative_to(plan.project_root).as_posix() or "."
                for package in plan.install_order
            )
            raise click.UsageError(
                f"Ui '{ui_name}' resolves to {resolved}, which is not inside the "
                f"target package '{plan.target.name}' or any of its workspace "
                f"dependencies. Only these directories are copied into the "
                f"container: {copied_dirs}. If this file lives in another "
                f"workspace package, add it as a dependency of "
                f"'{plan.target.name}' with `{{ workspace = true }}` in "
                "[tool.uv.sources]."
            )

        ui[ui_name] = container_path.as_posix()


def python_config_to_docker_uv_lock(
    config_path: pathlib.Path,
    config: Config,
    base_image: str,
    api_version: str | None = None,
    *,
    build_tools_to_uninstall: tuple[str, ...] | None,
) -> tuple[str, dict[str, str]]:
    from langgraph_cli.config import (
        DEFAULT_NODE_VERSION,
        _build_python_install_commands,
        _build_runtime_env_vars,
        _get_node_pm_install_cmd,
        _get_pip_cleanup_lines,
        _image_supports_uv,
        docker_tag,
    )

    if not _image_supports_uv(base_image):
        raise ValueError(
            "source.kind 'uv' requires a base image with uv support "
            "(langchain/langgraph-api >= 0.2.47)"
        )

    config_root = config_path.parent.resolve()
    install_cmd = "uv pip install --system"
    _, global_reqs_pip_install, pip_config_file_str = _build_python_install_commands(
        config, install_cmd
    )
    plan = _plan_uv_lock_workspace(config_path, config)

    _update_uv_lock_graph_paths(config_path, config, plan)
    for section, key in [
        ("auth", "path"),
        ("encryption", "path"),
        ("checkpointer", "path"),
        ("http", "app"),
    ]:
        _update_uv_lock_component_path(
            config_path,
            config,
            plan,
            section=section,
            key=key,
            label=f"{section}.{key}",
        )
    _update_uv_lock_ui_paths(config_path, config, plan)

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
    env_vars = _build_runtime_env_vars(config)

    image_str = docker_tag(config, base_image, api_version)
    docker_plan = DockerBuildPlan(lines=[])

    if additional_contexts:
        docker_plan.add_raw("# syntax=docker/dockerfile:1.4")
        docker_plan.add_blank()

    docker_plan.add_instruction("FROM", image_str)
    docker_plan.add_blank()
    docker_plan.extend_nonempty(config["dockerfile_lines"])
    if config["dockerfile_lines"]:
        docker_plan.add_blank()

    if (config.get("ui") or config.get("node_version")) and plan.working_dir:
        docker_plan.add_instruction("RUN", "/storage/install-node.sh")
        docker_plan.add_blank()

    if pip_config_file_str:
        docker_plan.add_raw(pip_config_file_str)
        docker_plan.add_blank()

    docker_plan.add_raw("# -- Installing dependencies from uv.lock --")
    docker_plan.add_raw(
        copy_from_project_root(
            pathlib.PurePosixPath("pyproject.toml"),
            f"{uv_export_project_dir}/pyproject.toml",
        )
    )
    docker_plan.add_raw(
        copy_from_project_root(
            pathlib.PurePosixPath("uv.lock"),
            f"{uv_export_project_dir}/uv.lock",
        )
    )
    docker_plan.add_instruction("WORKDIR", uv_export_project_dir)
    docker_plan.add_instruction(
        "RUN",
        " ".join(
            [
                "uv export",
                f"--package {shlex.quote(plan.target.name)}",
                "--frozen",
                "--no-hashes",
                "--no-emit-project",
                "--no-emit-workspace",
                "-o uv_requirements.txt",
            ]
        ),
    )
    docker_plan.add_instruction(
        "RUN", f"{global_reqs_pip_install} -r uv_requirements.txt"
    )
    docker_plan.add_instruction("RUN", "rm -rf /tmp/uv_export")
    docker_plan.add_raw("# -- End of uv.lock dependencies install --")
    docker_plan.add_blank()

    for package in plan.install_order:
        package_label = package.root.relative_to(plan.project_root).as_posix() or "."
        docker_plan.add_raw(f"# -- Adding workspace package {package_label} --")
        for source, destination in _uv_lock_package_copy_items(package, plan):
            docker_plan.add_raw(copy_from_project_root(source, destination.as_posix()))
        docker_plan.add_instruction(
            "WORKDIR", plan.container_roots[package.root].as_posix()
        )
        docker_plan.add_instruction("RUN", f"{global_reqs_pip_install} --no-deps -e .")
        docker_plan.add_raw(f"# -- End of workspace package {package_label} --")
        docker_plan.add_blank()

    docker_plan.extend_nonempty(env_vars)
    if env_vars:
        docker_plan.add_blank()

    if (config.get("ui") or config.get("node_version")) and plan.working_dir:
        docker_plan.add_raw("# -- Installing JS dependencies --")
        docker_plan.add_instruction(
            "ENV", f"NODE_VERSION={config.get('node_version') or DEFAULT_NODE_VERSION}"
        )
        docker_plan.add_instruction("WORKDIR", plan.working_dir)
        docker_plan.add_instruction(
            "RUN",
            f"{_get_node_pm_install_cmd(plan.target_root)} && "
            "tsx /api/langgraph_api/js/build.mts",
        )
        docker_plan.add_raw("# -- End of JS dependencies install --")
        docker_plan.add_blank()

    docker_plan.add_raw(
        _get_pip_cleanup_lines(
            install_cmd=install_cmd,
            to_uninstall=build_tools_to_uninstall,
            pip_installer="uv",
        )
    )
    docker_plan.add_blank()
    if plan.working_dir:
        docker_plan.add_instruction("WORKDIR", plan.working_dir)

    return docker_plan.render(), additional_contexts
