use std::collections::HashSet;
use std::path::Path;

use indexmap::IndexMap;

use crate::config::docker_tag::docker_tag;
use crate::config::local_deps::{assemble_local_deps, LocalDeps};
use crate::config::path_rewrite::{
    update_auth_path, update_encryption_path, update_graph_paths, update_http_app_path,
};
use crate::config::schema::{Config, KeepPkgTools};
use crate::constants::{BUILD_TOOLS, DEFAULT_NODE_VERSION};

use regex::Regex;

/// Check if a base image supports uv.
fn image_supports_uv(base_image: &str) -> bool {
    if base_image == "langchain/langgraph-trial" {
        return false;
    }
    let re = Regex::new(r":(\d+(?:\.\d+)?(?:\.\d+)?)(?:-|$)").unwrap();
    match re.find(base_image) {
        None => true, // Default image supports it
        Some(m) => {
            let raw = &base_image[m.start() + 1..m.end()];
            let version_str = raw.trim_end_matches('-');
            let parts: Vec<u32> = version_str.split('.').filter_map(|p: &str| p.parse().ok()).collect();
            let version = (
                *parts.first().unwrap_or(&0),
                *parts.get(1).unwrap_or(&0),
                *parts.get(2).unwrap_or(&0),
            );
            version >= (0, 2, 47)
        }
    }
}

/// Get build tools to uninstall based on config.
fn get_build_tools_to_uninstall(config: &Config) -> Vec<String> {
    match &config.keep_pkg_tools {
        None => BUILD_TOOLS.iter().map(|s| s.to_string()).collect(),
        Some(KeepPkgTools::Bool(true)) => vec![],
        Some(KeepPkgTools::Bool(false)) => BUILD_TOOLS.iter().map(|s| s.to_string()).collect(),
        Some(KeepPkgTools::List(keep)) => {
            let keep_set: HashSet<&str> = keep.iter().map(|s| s.as_str()).collect();
            BUILD_TOOLS
                .iter()
                .filter(|t| !keep_set.contains(*t))
                .map(|s| s.to_string())
                .collect()
        }
    }
}

/// Generate pip cleanup lines for the Dockerfile.
fn get_pip_cleanup_lines(
    install_cmd: &str,
    to_uninstall: &[String],
    pip_installer: &str,
) -> String {
    let mut commands = vec![format!(
        "# -- Ensure user deps didn't inadvertently overwrite langgraph-api\n\
         RUN mkdir -p /api/langgraph_api /api/langgraph_runtime /api/langgraph_license && \\\n\
         touch /api/langgraph_api/__init__.py /api/langgraph_runtime/__init__.py /api/langgraph_license/__init__.py\n\
         RUN PYTHONDONTWRITEBYTECODE=1 {install_cmd} --no-cache-dir --no-deps -e /api\n\
         # -- End of ensuring user deps didn't inadvertently overwrite langgraph-api --\n\
         # -- Removing build deps from the final image ~<:===~~~ --"
    )];

    if !to_uninstall.is_empty() {
        let mut sorted = to_uninstall.to_vec();
        sorted.sort();
        let packs_str = sorted.join(" ");
        commands.push(format!("RUN pip uninstall -y {packs_str}"));

        let packages_rm: String = to_uninstall
            .iter()
            .map(|p| format!("/usr/local/lib/python*/site-packages/{p}*"))
            .collect::<Vec<_>>()
            .join(" ");
        let mut rm_cmd = format!("RUN rm -rf {packages_rm}");
        if to_uninstall.contains(&"pip".to_string()) {
            rm_cmd.push_str(" && find /usr/local/bin -name \"pip*\" -delete || true");
        }
        commands.push(rm_cmd);

        let wolfi_rm: String = to_uninstall
            .iter()
            .map(|p| format!("/usr/lib/python*/site-packages/{p}*"))
            .collect::<Vec<_>>()
            .join(" ");
        let mut wolfi_cmd = format!("RUN rm -rf {wolfi_rm}");
        if to_uninstall.contains(&"pip".to_string()) {
            wolfi_cmd.push_str(" && find /usr/bin -name \"pip*\" -delete || true");
        }
        commands.push(wolfi_cmd);

        if pip_installer == "uv" {
            commands.push(format!(
                "RUN uv pip uninstall --system {packs_str} && rm /usr/bin/uv /usr/bin/uvx"
            ));
        }
    } else if pip_installer == "uv" {
        commands.push(
            "RUN rm /usr/bin/uv /usr/bin/uvx\n# -- End of build deps removal --".to_string(),
        );
    }

    commands.join("\n")
}

/// Detect the Node.js package manager install command.
fn get_node_pm_install_cmd(config_path: &Path, _config: &Config) -> String {
    let parent = config_path.parent().unwrap();

    let test_file = |name: &str| -> bool { parent.join(name).is_file() };

    let npm = test_file("package-lock.json");
    let yarn = test_file("yarn.lock");
    let pnpm = test_file("pnpm-lock.yaml");
    let bun = test_file("bun.lockb");

    if yarn {
        "yarn install --frozen-lockfile".to_string()
    } else if pnpm {
        "pnpm i --frozen-lockfile".to_string()
    } else if npm {
        "npm ci".to_string()
    } else if bun {
        "bun i".to_string()
    } else {
        // Try to detect from package.json
        let pkg_manager = get_pkg_manager_name(parent);
        match pkg_manager.as_deref() {
            Some("yarn") => "yarn install".to_string(),
            Some("pnpm") => "pnpm i".to_string(),
            Some("bun") => "bun i".to_string(),
            _ => "npm i".to_string(),
        }
    }
}

fn get_pkg_manager_name(dir: &Path) -> Option<String> {
    let pkg_path = dir.join("package.json");
    let content = std::fs::read_to_string(pkg_path).ok()?;
    let pkg: serde_json::Value = serde_json::from_str(&content).ok()?;

    // Check packageManager field
    if let Some(pm) = pkg.get("packageManager").and_then(|v| v.as_str()) {
        let name = pm.trim_start_matches('^').split('@').next()?;
        return Some(name.to_string());
    }

    // Check devEngines.packageManager.name
    if let Some(name) = pkg
        .get("devEngines")
        .and_then(|de| de.get("packageManager"))
        .and_then(|pm| pm.get("name"))
        .and_then(|n| n.as_str())
    {
        return Some(name.to_string());
    }

    None
}

/// Generate a Dockerfile for a Python configuration.
pub fn python_config_to_docker(
    config_path: &Path,
    config: &mut Config,
    base_image: &str,
    api_version: Option<&str>,
    escape_variables: bool,
) -> Result<(String, IndexMap<String, String>), String> {
    let build_tools_to_uninstall = get_build_tools_to_uninstall(config);

    let pip_installer_raw = config
        .pip_installer
        .as_deref()
        .unwrap_or("auto")
        .to_string();
    let pip_installer: String = if pip_installer_raw == "auto" {
        if image_supports_uv(base_image) {
            "uv".to_string()
        } else {
            "pip".to_string()
        }
    } else {
        pip_installer_raw
    };

    let install_cmd = if pip_installer == "uv" {
        "uv pip install --system"
    } else {
        "pip install"
    };

    // Configure pip
    let local_reqs_pip_install = {
        let base = format!("PYTHONDONTWRITEBYTECODE=1 {install_cmd} --no-cache-dir -c /api/constraints.txt");
        if config.pip_config_file.is_some() {
            format!("PIP_CONFIG_FILE=/pipconfig.txt {base}")
        } else {
            base
        }
    };
    let global_reqs_pip_install = {
        let base = format!("PYTHONDONTWRITEBYTECODE=1 {install_cmd} --no-cache-dir -c /api/constraints.txt");
        if config.pip_config_file.is_some() {
            format!("PIP_CONFIG_FILE=/pipconfig.txt {base}")
        } else {
            base
        }
    };

    let pip_config_file_str = config
        .pip_config_file
        .as_ref()
        .map(|f| format!("ADD {f} /pipconfig.txt"))
        .unwrap_or_default();

    // Collect dependencies (owned to avoid borrowing config across mutable calls)
    let pypi_deps: Vec<String> = config
        .dependencies
        .iter()
        .filter(|dep| !dep.starts_with('.'))
        .cloned()
        .collect();

    let local_deps = assemble_local_deps(config_path, config)?;

    // Rewrite paths
    update_graph_paths(config_path, config, &local_deps)?;
    update_auth_path(config_path, config, &local_deps)?;
    update_encryption_path(config_path, config, &local_deps)?;
    update_http_app_path(config_path, config, &local_deps)?;

    let pip_pkgs_str = if !pypi_deps.is_empty() {
        format!("RUN {local_reqs_pip_install} {}", pypi_deps.join(" "))
    } else {
        String::new()
    };

    // Build pip requirements string
    let pip_reqs_str = build_pip_reqs_str(&local_deps, config_path, &local_reqs_pip_install);

    // Build faux packages string
    let faux_pkgs_str = build_faux_pkgs_str(&local_deps, config_path);

    // Build local packages string
    let local_pkgs_str = build_local_pkgs_str(&local_deps, config_path);

    // Install Node.js if needed
    let install_node_str = if (config.ui.is_some() || config.node_version.is_some())
        && local_deps.working_dir.is_some()
    {
        "RUN /storage/install-node.sh".to_string()
    } else {
        String::new()
    };

    // Combine install sections
    let installs: Vec<&str> = [
        &install_node_str,
        &pip_config_file_str,
        &pip_pkgs_str,
        &pip_reqs_str,
        &local_pkgs_str,
        &faux_pkgs_str,
    ]
    .iter()
    .filter(|s| !s.is_empty())
    .map(|s| s.as_str())
    .collect();
    let installs = installs.join("\n\n");

    // Environment variables
    let mut env_vars = Vec::new();

    if let Some(ref store) = config.store {
        env_vars.push(format!(
            "ENV LANGGRAPH_STORE='{}'",
            serde_json::to_string(store).unwrap()
        ));
    }
    if let Some(ref auth) = config.auth {
        env_vars.push(format!(
            "ENV LANGGRAPH_AUTH='{}'",
            serde_json::to_string(auth).unwrap()
        ));
    }
    if let Some(ref encryption) = config.encryption {
        env_vars.push(format!(
            "ENV LANGGRAPH_ENCRYPTION='{}'",
            serde_json::to_string(encryption).unwrap()
        ));
    }
    if let Some(ref http) = config.http {
        env_vars.push(format!(
            "ENV LANGGRAPH_HTTP='{}'",
            serde_json::to_string(http).unwrap()
        ));
    }
    if let Some(ref webhooks) = config.webhooks {
        env_vars.push(format!(
            "ENV LANGGRAPH_WEBHOOKS='{}'",
            serde_json::to_string(webhooks).unwrap()
        ));
    }
    if let Some(ref checkpointer) = config.checkpointer {
        env_vars.push(format!(
            "ENV LANGGRAPH_CHECKPOINTER='{}'",
            serde_json::to_string(checkpointer).unwrap()
        ));
    }
    if let Some(ref ui) = config.ui {
        env_vars.push(format!(
            "ENV LANGGRAPH_UI='{}'",
            serde_json::to_string(ui).unwrap()
        ));
    }
    if let Some(ref ui_config) = config.ui_config {
        env_vars.push(format!(
            "ENV LANGGRAPH_UI_CONFIG='{}'",
            serde_json::to_string(ui_config).unwrap()
        ));
    }
    env_vars.push(format!(
        "ENV LANGSERVE_GRAPHS='{}'",
        serde_json::to_string(&config.graphs).unwrap()
    ));

    // JS install
    let js_inst_str =
        if (config.ui.is_some() || config.node_version.is_some()) && local_deps.working_dir.is_some()
        {
            let default_node_ver = DEFAULT_NODE_VERSION.to_string();
            let node_version = config
                .node_version
                .as_deref()
                .unwrap_or(&default_node_ver);
            let install_cmd_node = get_node_pm_install_cmd(config_path, config);
            let wd = local_deps.working_dir.as_ref().unwrap();
            format!(
                "# -- Installing JS dependencies --\n\
                 ENV NODE_VERSION={node_version}\n\
                 RUN cd {wd} && {install_cmd_node} && tsx /api/langgraph_api/js/build.mts\n\
                 # -- End of JS dependencies install --"
            )
        } else {
            String::new()
        };

    let image_str = docker_tag(config, Some(base_image), api_version);

    // Build Dockerfile
    let mut docker_file_contents = Vec::new();

    // Syntax directive for additional contexts
    if !local_deps.additional_contexts.is_empty() {
        docker_file_contents.push("# syntax=docker/dockerfile:1.4".to_string());
        docker_file_contents.push(String::new());
    }

    let dep_vname = if escape_variables { "$$dep" } else { "$dep" };

    docker_file_contents.extend_from_slice(&[
        format!("FROM {image_str}"),
        String::new(),
        config.dockerfile_lines.join("\n"),
        String::new(),
        installs,
        String::new(),
        "# -- Installing all local dependencies --".to_string(),
        format!(
            "RUN for dep in /deps/*; do \\\n\
             {0}    echo \"Installing {dep_vname}\"; \\\n\
             {0}    if [ -d \"{dep_vname}\" ]; then \\\n\
             {0}        echo \"Installing {dep_vname}\"; \\\n\
             {0}        (cd \"{dep_vname}\" && {global_reqs_pip_install} -e .); \\\n\
             {0}    fi; \\\n\
             {0}done",
            "    "
        ),
        "# -- End of local dependencies install --".to_string(),
        env_vars.join("\n"),
        String::new(),
        js_inst_str,
        String::new(),
        get_pip_cleanup_lines(install_cmd, &build_tools_to_uninstall, &pip_installer),
        String::new(),
        local_deps
            .working_dir
            .as_ref()
            .map(|wd| format!("WORKDIR {wd}"))
            .unwrap_or_default(),
    ]);

    // Build additional contexts map
    let mut additional_contexts = IndexMap::new();
    for p in &local_deps.additional_contexts {
        if let Some((_, name)) = local_deps.real_pkgs.get(p) {
            additional_contexts.insert(name.clone(), p.to_string_lossy().to_string());
        } else if local_deps.faux_pkgs.contains_key(p) {
            let name = format!("outer-{}", p.file_name().unwrap().to_string_lossy());
            additional_contexts.insert(name, p.to_string_lossy().to_string());
        }
    }

    Ok((docker_file_contents.join("\n"), additional_contexts))
}

/// Generate a Dockerfile for a Node.js configuration.
pub fn node_config_to_docker(
    config_path: &Path,
    config: &mut Config,
    base_image: &str,
    api_version: Option<&str>,
    install_command: Option<&str>,
    build_command: Option<&str>,
    build_context: Option<&str>,
) -> Result<(String, IndexMap<String, String>), String> {
    let (faux_path, container_root) = if let Some(bc) = build_context {
        let relative_workdir = calculate_relative_workdir(config_path, bc)?;
        let container_name = Path::new(bc)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let fp = if relative_workdir.is_empty() {
            format!("/deps/{container_name}")
        } else {
            format!("/deps/{container_name}/{relative_workdir}")
        };
        (fp, Some(format!("/deps/{container_name}")))
    } else {
        let config_dir_name = config_path
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        (format!("/deps/{config_dir_name}"), None)
    };

    let install_cmd = install_command
        .map(|s| s.to_string())
        .unwrap_or_else(|| get_node_pm_install_cmd(config_path, config));

    let image_str = docker_tag(config, Some(base_image), api_version);

    // Environment variables
    let mut env_vars = Vec::new();
    if let Some(ref store) = config.store {
        env_vars.push(format!(
            "ENV LANGGRAPH_STORE='{}'",
            serde_json::to_string(store).unwrap()
        ));
    }
    if let Some(ref auth) = config.auth {
        env_vars.push(format!(
            "ENV LANGGRAPH_AUTH='{}'",
            serde_json::to_string(auth).unwrap()
        ));
    }
    if let Some(ref encryption) = config.encryption {
        env_vars.push(format!(
            "ENV LANGGRAPH_ENCRYPTION='{}'",
            serde_json::to_string(encryption).unwrap()
        ));
    }
    if let Some(ref http) = config.http {
        env_vars.push(format!(
            "ENV LANGGRAPH_HTTP='{}'",
            serde_json::to_string(http).unwrap()
        ));
    }
    if let Some(ref webhooks) = config.webhooks {
        env_vars.push(format!(
            "ENV LANGGRAPH_WEBHOOKS='{}'",
            serde_json::to_string(webhooks).unwrap()
        ));
    }
    if let Some(ref checkpointer) = config.checkpointer {
        env_vars.push(format!(
            "ENV LANGGRAPH_CHECKPOINTER='{}'",
            serde_json::to_string(checkpointer).unwrap()
        ));
    }
    if let Some(ref ui) = config.ui {
        env_vars.push(format!(
            "ENV LANGGRAPH_UI='{}'",
            serde_json::to_string(ui).unwrap()
        ));
    }
    if let Some(ref ui_config) = config.ui_config {
        env_vars.push(format!(
            "ENV LANGGRAPH_UI_CONFIG='{}'",
            serde_json::to_string(ui_config).unwrap()
        ));
    }
    env_vars.push(format!(
        "ENV LANGSERVE_GRAPHS='{}'",
        serde_json::to_string(&config.graphs).unwrap()
    ));

    let (install_step, build_step) = if let Some(_bc) = build_context {
        let cr = container_root.as_ref().unwrap();
        let inst = format!("RUN cd {cr} && {install_cmd}");
        let build = if let Some(bc) = build_command {
            format!("RUN cd {faux_path} && {bc}")
        } else {
            "RUN (test ! -f /api/langgraph_api/js/build.mts && echo \"Prebuild script not found, skipping\") || tsx /api/langgraph_api/js/build.mts".to_string()
        };
        (inst, build)
    } else {
        let inst = format!("RUN cd {faux_path} && {install_cmd}");
        let build = "RUN (test ! -f /api/langgraph_api/js/build.mts && echo \"Prebuild script not found, skipping\") || tsx /api/langgraph_api/js/build.mts".to_string();
        (inst, build)
    };

    let add_target = if build_context.is_some() {
        container_root.as_ref().unwrap().clone()
    } else {
        faux_path.clone()
    };

    let docker_file_contents = vec![
        format!("FROM {image_str}"),
        String::new(),
        config.dockerfile_lines.join("\n"),
        String::new(),
        format!("ADD . {add_target}"),
        String::new(),
        install_step,
        String::new(),
        env_vars.join("\n"),
        String::new(),
        format!("WORKDIR {faux_path}"),
        String::new(),
        build_step,
    ];

    Ok((docker_file_contents.join("\n"), IndexMap::new()))
}

/// Main entry point for Dockerfile generation.
pub fn config_to_docker(
    config_path: &Path,
    config: &mut Config,
    base_image: Option<&str>,
    api_version: Option<&str>,
    install_command: Option<&str>,
    build_command: Option<&str>,
    build_context: Option<&str>,
    escape_variables: bool,
) -> Result<(String, IndexMap<String, String>), String> {
    let base_image = base_image
        .map(|s| s.to_string())
        .unwrap_or_else(|| crate::config::docker_tag::default_base_image(config));

    if config.node_version.is_some() && config.python_version.is_none() {
        node_config_to_docker(
            config_path,
            config,
            &base_image,
            api_version,
            install_command,
            build_command,
            build_context,
        )
    } else {
        python_config_to_docker(config_path, config, &base_image, api_version, escape_variables)
    }
}

fn calculate_relative_workdir(config_path: &Path, build_context: &str) -> Result<String, String> {
    let config_dir = config_path.parent().unwrap().canonicalize().map_err(|e| {
        format!(
            "Could not resolve config directory: {e}"
        )
    })?;
    let build_context_path = Path::new(build_context).canonicalize().map_err(|e| {
        format!("Could not resolve build context: {e}")
    })?;

    config_dir
        .strip_prefix(&build_context_path)
        .map(|p| {
            let s = p.to_string_lossy().to_string();
            if s == "." { String::new() } else { s }
        })
        .map_err(|_| {
            format!(
                "Configuration file {} is not under the build context {}. \
                 Please run the command from a directory that contains your langgraph.json file.",
                config_path.display(),
                build_context
            )
        })
}

// Helper functions to build Dockerfile sections

fn build_pip_reqs_str(local_deps: &LocalDeps, config_path: &Path, pip_install: &str) -> String {
    if local_deps.pip_reqs.is_empty() {
        return String::new();
    }

    let config_parent = config_path.parent().unwrap();
    let mut lines = Vec::new();

    for (reqpath, destpath) in &local_deps.pip_reqs {
        if local_deps
            .additional_contexts
            .iter()
            .any(|ac| reqpath.starts_with(ac))
        {
            let name = reqpath.parent().unwrap().file_name().unwrap().to_string_lossy();
            lines.push(format!(
                "COPY --from=outer-{name} requirements.txt {destpath}"
            ));
        } else if let Ok(rel) = reqpath.strip_prefix(config_parent) {
            lines.push(format!(
                "ADD {} {destpath}",
                rel.to_string_lossy().replace('\\', "/")
            ));
        }
    }

    let req_args: Vec<String> = local_deps
        .pip_reqs
        .iter()
        .map(|(_, dest)| format!("-r {dest}"))
        .collect();
    lines.push(format!("RUN {pip_install} {}", req_args.join(" ")));

    format!(
        "# -- Installing local requirements --\n{}\n# -- End of local requirements install --",
        lines.join("\n")
    )
}

fn build_faux_pkgs_str(local_deps: &LocalDeps, config_path: &Path) -> String {
    if local_deps.faux_pkgs.is_empty() {
        return String::new();
    }

    let _config_parent = config_path.parent().unwrap();
    let mut sections = Vec::new();

    for (fullpath, (relpath, destpath)) in &local_deps.faux_pkgs {
        let name = fullpath.file_name().unwrap().to_string_lossy();

        let add_line = if local_deps.additional_contexts.contains(fullpath) {
            format!(
                "# -- Adding non-package dependency {name} --\n\
                 COPY --from=outer-{name} . {destpath}"
            )
        } else {
            format!(
                "# -- Adding non-package dependency {name} --\n\
                 ADD {relpath} {destpath}"
            )
        };

        let section = format!(
            "{add_line}\n\
             RUN set -ex && \\\\\n\
             {s}for line in '[project]' \\\\\n\
             {s}            'name = \"{name}\"' \\\\\n\
             {s}            'version = \"0.1\"' \\\\\n\
             {s}            '[tool.setuptools.package-data]' \\\\\n\
             {s}            '\"*\" = [\"**/*\"]' \\\\\n\
             {s}            '[build-system]' \\\\\n\
             {s}            'requires = [\"setuptools>=61\"]' \\\\\n\
             {s}            'build-backend = \"setuptools.build_meta\"'; do \\\\\n\
             {s}    echo \"$line\" >> /deps/outer-{name}/pyproject.toml; \\\\\n\
             {s}done\n\
             # -- End of non-package dependency {name} --",
            s = "    "
        );
        sections.push(section);
    }

    sections.join("\n\n")
}

fn build_local_pkgs_str(local_deps: &LocalDeps, config_path: &Path) -> String {
    if local_deps.real_pkgs.is_empty() {
        return String::new();
    }

    let _config_parent = config_path.parent().unwrap();
    let mut lines = Vec::new();

    for (fullpath, (relpath, name)) in &local_deps.real_pkgs {
        if local_deps.additional_contexts.contains(fullpath) {
            lines.push(format!(
                "# -- Adding local package {relpath} --\n\
                 COPY --from={name} . /deps/{name}\n\
                 # -- End of local package {relpath} --"
            ));
        } else {
            lines.push(format!(
                "# -- Adding local package {relpath} --\n\
                 ADD {relpath} /deps/{name}\n\
                 # -- End of local package {relpath} --"
            ));
        }
    }

    lines.join("\n")
}

/// Generate compose YAML fragment for config (used by the up command).
pub fn config_to_compose(
    config_path: &Path,
    config: &mut Config,
    base_image: Option<&str>,
    api_version: Option<&str>,
    image: Option<&str>,
    watch: bool,
) -> Result<String, String> {
    let base_image = base_image
        .map(|s| s.to_string())
        .unwrap_or_else(|| crate::config::docker_tag::default_base_image(config));

    let env_vars_str = match &config.env {
        Some(crate::config::schema::EnvConfig::Dict(d)) => d
            .iter()
            .map(|(k, v)| format!("            {k}: \"{v}\""))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    };

    let env_file_str = match &config.env {
        Some(crate::config::schema::EnvConfig::File(f)) => format!("env_file: {f}"),
        _ => String::new(),
    };

    let watch_str = if watch {
        let dependencies = if config.dependencies.is_empty() {
            vec![".".to_string()]
        } else {
            config.dependencies.clone()
        };
        let config_name = config_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let mut watch_paths = vec![config_name];
        for dep in &dependencies {
            if dep.starts_with('.') {
                watch_paths.push(dep.clone());
            }
        }
        let watch_actions: Vec<String> = watch_paths
            .iter()
            .map(|p| format!("                - path: {p}\n                  action: rebuild"))
            .collect();
        format!(
            "\n        develop:\n            watch:\n{}\n",
            watch_actions.join("\n")
        )
    } else {
        String::new()
    };

    if image.is_some() {
        return Ok(format!(
            "\n{env_vars_str}\n        {env_file_str}\n        {watch_str}\n"
        ));
    }

    let (dockerfile, additional_contexts) = config_to_docker(
        config_path,
        config,
        Some(&base_image),
        api_version,
        None,
        None,
        None,
        true, // escape_variables
    )?;

    let additional_contexts_str = if !additional_contexts.is_empty() {
        let lines: Vec<String> = additional_contexts
            .iter()
            .map(|(name, path)| format!("                - {name}: {path}"))
            .collect();
        format!(
            "\n            additional_contexts:\n{}",
            lines.join("\n")
        )
    } else {
        String::new()
    };

    // Indent dockerfile for inline use
    let indented_dockerfile = dockerfile
        .lines()
        .map(|line| format!("                {line}"))
        .collect::<Vec<_>>()
        .join("\n");

    Ok(format!(
        "\n{env_vars_str}\n        {env_file_str}\n        pull_policy: build\n        build:\n            context: .{additional_contexts_str}\n            dockerfile_inline: |\n{indented_dockerfile}\n        {watch_str}\n"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::validate_config;
    use crate::config::schema::Config;
    use crate::util::clean_empty_lines;
    use serde_json::json;

    /// Helper: get the path to the test config file.
    fn config_path() -> std::path::PathBuf {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        std::path::Path::new(manifest_dir).join("tests/unit_tests/test_config.json")
    }

    /// Helper: build a validated Config from a JSON value.
    fn make_config(val: serde_json::Value) -> Config {
        let config: Config = serde_json::from_value(val).expect("Failed to deserialize config");
        validate_config(config).expect("Failed to validate config")
    }

    // -- test_config_to_docker_nodejs --
    #[test]
    fn test_config_to_docker_nodejs() {
        let cp = config_path();
        let mut config = make_config(json!({
            "node_version": "20",
            "graphs": {"agent": "./graphs/agent.js:graph"},
            "dockerfile_lines": ["ARG meow", "ARG foo"],
            "auth": {"path": "./graphs/auth.mts:auth"},
            "ui": {"agent": "./graphs/agent.ui.jsx"},
            "ui_config": {"shared": ["nuqs"]}
        }));
        let (actual, additional_contexts) = config_to_docker(
            &cp,
            &mut config,
            Some("langchain/langgraphjs-api"),
            None,
            None,
            None,
            None,
            false,
        )
        .expect("config_to_docker failed");

        let expected = "\
FROM langchain/langgraphjs-api:20
ARG meow
ARG foo
ADD . /deps/unit_tests
RUN cd /deps/unit_tests && npm i
ENV LANGGRAPH_AUTH='{\"path\":\"./graphs/auth.mts:auth\"}'
ENV LANGGRAPH_UI='{\"agent\":\"./graphs/agent.ui.jsx\"}'
ENV LANGGRAPH_UI_CONFIG='{\"shared\":[\"nuqs\"]}'
ENV LANGSERVE_GRAPHS='{\"agent\":\"./graphs/agent.js:graph\"}'
WORKDIR /deps/unit_tests
RUN (test ! -f /api/langgraph_api/js/build.mts && echo \"Prebuild script not found, skipping\") || tsx /api/langgraph_api/js/build.mts";

        assert_eq!(clean_empty_lines(&actual), expected);
        assert!(additional_contexts.is_empty());
    }

    // -- test_config_to_docker_nodejs_internal_docker_tag --
    #[test]
    fn test_config_to_docker_nodejs_internal_docker_tag() {
        let cp = config_path();
        let mut config = make_config(json!({
            "node_version": "20",
            "graphs": {"agent": "./graphs/agent.js:graph"},
            "dockerfile_lines": ["ARG meow", "ARG foo"],
            "auth": {"path": "./graphs/auth.mts:auth"},
            "ui": {"agent": "./graphs/agent.ui.jsx"},
            "ui_config": {"shared": ["nuqs"]},
            "_INTERNAL_docker_tag": "my-tag"
        }));
        let (actual, additional_contexts) = config_to_docker(
            &cp,
            &mut config,
            Some("langchain/langgraphjs-api"),
            None,
            None,
            None,
            None,
            false,
        )
        .expect("config_to_docker failed");

        let expected = "\
FROM langchain/langgraphjs-api:my-tag
ARG meow
ARG foo
ADD . /deps/unit_tests
RUN cd /deps/unit_tests && npm i
ENV LANGGRAPH_AUTH='{\"path\":\"./graphs/auth.mts:auth\"}'
ENV LANGGRAPH_UI='{\"agent\":\"./graphs/agent.ui.jsx\"}'
ENV LANGGRAPH_UI_CONFIG='{\"shared\":[\"nuqs\"]}'
ENV LANGSERVE_GRAPHS='{\"agent\":\"./graphs/agent.js:graph\"}'
WORKDIR /deps/unit_tests
RUN (test ! -f /api/langgraph_api/js/build.mts && echo \"Prebuild script not found, skipping\") || tsx /api/langgraph_api/js/build.mts";

        assert_eq!(clean_empty_lines(&actual), expected);
        assert!(additional_contexts.is_empty());
    }

    // -- test_config_to_docker_invalid_inputs --
    #[test]
    fn test_config_to_docker_invalid_missing_dependency() {
        let cp = config_path();
        let mut config = make_config(json!({
            "dependencies": ["./missing"],
            "graphs": {"agent": "tests/unit_tests/agent.py:graph"}
        }));
        let result = config_to_docker(
            &cp,
            &mut config,
            Some("langchain/langgraph-api"),
            None,
            None,
            None,
            None,
            false,
        );
        assert!(result.is_err(), "Expected error for missing dependency");
        let err = result.unwrap_err();
        assert!(
            err.contains("Could not find local dependency"),
            "Error should mention missing dependency, got: {err}"
        );
    }

    #[test]
    fn test_config_to_docker_invalid_missing_module() {
        let cp = config_path();
        let mut config = make_config(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./missing_agent.py:graph"}
        }));
        let result = config_to_docker(
            &cp,
            &mut config,
            Some("langchain/langgraph-api"),
            None,
            None,
            None,
            None,
            false,
        );
        assert!(result.is_err(), "Expected error for missing module");
        let err = result.unwrap_err();
        assert!(
            err.contains("Could not find local module"),
            "Error should mention missing module, got: {err}"
        );
    }

    // -- test_config_to_docker_python_encryption --
    #[test]
    fn test_config_to_docker_python_encryption() {
        let config = make_config(json!({
            "python_version": "3.11",
            "graphs": {"agent": "./agent.py:graph"},
            "dependencies": ["."],
            "encryption": {"path": "./encryption.py:encryption"}
        }));
        assert!(config.encryption.is_some());
        assert_eq!(
            config.encryption.as_ref().unwrap().path.as_deref(),
            Some("./encryption.py:encryption")
        );
    }

    // -- test_config_to_docker_python_encryption_bad_path --
    #[test]
    fn test_config_to_docker_python_encryption_bad_path() {
        let raw_config: Config = serde_json::from_value(json!({
            "python_version": "3.11",
            "graphs": {"agent": "./agent.py:graph"},
            "dependencies": ["."],
            "encryption": {"path": "./encryption.py"}
        }))
        .expect("Failed to deserialize config");

        let result = validate_config(raw_config);
        assert!(result.is_err(), "Expected error for bad encryption path");
        let err = result.unwrap_err();
        assert!(
            err.contains("Invalid encryption.path format"),
            "Error should mention invalid encryption.path format, got: {err}"
        );
    }

    // -- test_config_to_docker_no_webhooks --
    #[test]
    fn test_config_to_docker_no_webhooks() {
        let cp = config_path();
        let mut config = make_config(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"}
        }));
        let (dockerfile, _) = config_to_docker(
            &cp,
            &mut config,
            Some("langchain/langgraph-api"),
            None,
            None,
            None,
            None,
            false,
        )
        .expect("config_to_docker failed");

        assert!(
            !dockerfile.contains("ENV LANGGRAPH_WEBHOOKS="),
            "Dockerfile should not contain LANGGRAPH_WEBHOOKS when no webhooks configured"
        );
    }

    // -- test_config_to_docker_with_api_version --
    #[test]
    fn test_config_to_docker_with_api_version_python() {
        let cp = config_path();
        let mut config = make_config(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"}
        }));
        let (actual, _) = config_to_docker(
            &cp,
            &mut config,
            Some("langchain/langgraph-api"),
            Some("0.2.74"),
            None,
            None,
            None,
            false,
        )
        .expect("config_to_docker failed");

        let first_line = actual.lines().next().unwrap();
        assert_eq!(
            first_line,
            "FROM langchain/langgraph-api:0.2.74-py3.11",
            "Python config with api_version should use versioned FROM"
        );
    }

    #[test]
    fn test_config_to_docker_with_api_version_nodejs() {
        let cp = config_path();
        let mut config = make_config(json!({
            "node_version": "20",
            "graphs": {"agent": "./graphs/agent.js:graph"}
        }));
        let (actual, _) = config_to_docker(
            &cp,
            &mut config,
            Some("langchain/langgraphjs-api"),
            Some("0.2.74"),
            None,
            None,
            None,
            false,
        )
        .expect("config_to_docker failed");

        let first_line = actual.lines().next().unwrap();
        assert_eq!(
            first_line,
            "FROM langchain/langgraphjs-api:0.2.74-node20",
            "Node.js config with api_version should use versioned FROM"
        );
    }

    // -- test_config_to_docker_pip_installer --
    #[test]
    fn test_config_to_docker_pip_installer_auto_uv() {
        // auto + UV-supporting image (0.2.47) -> uv pip install
        let cp = config_path();
        let mut config = make_config(json!({
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./graphs/agent.py:graph"},
            "pip_installer": "auto"
        }));
        let (docker, _) = config_to_docker(
            &cp,
            &mut config,
            Some("langchain/langgraph-api:0.2.47"),
            None,
            None,
            None,
            None,
            false,
        )
        .expect("config_to_docker failed");

        assert!(
            docker.contains("uv pip install --system"),
            "auto + UV image should use uv pip install"
        );
        assert!(
            docker.contains("rm /usr/bin/uv /usr/bin/uvx"),
            "auto + UV image should remove uv after install"
        );
    }

    #[test]
    fn test_config_to_docker_pip_installer_explicit_pip() {
        // explicit pip + UV-supporting image -> pip install (not uv)
        let cp = config_path();
        let mut config = make_config(json!({
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./graphs/agent.py:graph"},
            "pip_installer": "pip"
        }));
        let (docker, _) = config_to_docker(
            &cp,
            &mut config,
            Some("langchain/langgraph-api:0.2.47"),
            None,
            None,
            None,
            None,
            false,
        )
        .expect("config_to_docker failed");

        assert!(
            !docker.contains("uv pip install --system"),
            "explicit pip should not use uv"
        );
        assert!(
            docker.contains("pip install"),
            "explicit pip should use pip install"
        );
        assert!(
            !docker.contains("rm /usr/bin/uv"),
            "explicit pip should not remove uv"
        );
    }

    #[test]
    fn test_config_to_docker_pip_installer_explicit_uv() {
        // explicit uv + UV-supporting image -> uv pip install
        let cp = config_path();
        let mut config = make_config(json!({
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./graphs/agent.py:graph"},
            "pip_installer": "uv"
        }));
        let (docker, _) = config_to_docker(
            &cp,
            &mut config,
            Some("langchain/langgraph-api:0.2.47"),
            None,
            None,
            None,
            None,
            false,
        )
        .expect("config_to_docker failed");

        assert!(
            docker.contains("uv pip install --system"),
            "explicit uv should use uv pip install"
        );
        assert!(
            docker.contains("rm /usr/bin/uv /usr/bin/uvx"),
            "explicit uv should remove uv after install"
        );
    }

    #[test]
    fn test_config_to_docker_pip_installer_auto_old_image() {
        // auto + older image (0.2.46) -> pip install (not uv)
        let cp = config_path();
        let mut config = make_config(json!({
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./graphs/agent.py:graph"},
            "pip_installer": "auto"
        }));
        let (docker, _) = config_to_docker(
            &cp,
            &mut config,
            Some("langchain/langgraph-api:0.2.46"),
            None,
            None,
            None,
            None,
            false,
        )
        .expect("config_to_docker failed");

        assert!(
            !docker.contains("uv pip install --system"),
            "auto + old image should not use uv"
        );
        assert!(
            docker.contains("pip install"),
            "auto + old image should use pip install"
        );
        assert!(
            !docker.contains("rm /usr/bin/uv"),
            "auto + old image should not remove uv"
        );
    }

    #[test]
    fn test_config_to_docker_pip_installer_default_auto() {
        // missing pip_installer defaults to auto -> uv on new image
        let cp = config_path();
        let mut config = make_config(json!({
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./graphs/agent.py:graph"}
        }));
        let (docker, _) = config_to_docker(
            &cp,
            &mut config,
            Some("langchain/langgraph-api:0.2.47"),
            None,
            None,
            None,
            None,
            false,
        )
        .expect("config_to_docker failed");

        assert!(
            docker.contains("uv pip install --system"),
            "default (auto) + new image should use uv pip install"
        );
    }

    // -- test_config_retain_build_tools --
    #[test]
    fn test_config_retain_build_tools_keep_all() {
        // keep_pkg_tools=true -> no pip uninstall, no rm of build tools
        let cp = config_path();
        let mut config = make_config(json!({
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./graphs/agent.py:graph"},
            "keep_pkg_tools": true
        }));
        let (docker, _) = config_to_docker(
            &cp,
            &mut config,
            Some("langchain/langgraph-api:0.2.47"),
            None,
            None,
            None,
            None,
            false,
        )
        .expect("config_to_docker failed");

        for tool in crate::constants::BUILD_TOOLS {
            let pattern = format!("/usr/local/lib/python*/site-packages/{tool}*");
            assert!(
                !docker.contains(&pattern),
                "keep_pkg_tools=true should not contain rm of {tool}"
            );
        }
        assert!(
            !docker.contains("RUN pip uninstall -y pip setuptools wheel"),
            "keep_pkg_tools=true should not uninstall build tools"
        );
    }

    #[test]
    fn test_config_retain_build_tools_keep_none() {
        // keep_pkg_tools=false -> pip uninstall all build tools, rm all
        let cp = config_path();
        let mut config = make_config(json!({
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./graphs/agent.py:graph"},
            "keep_pkg_tools": false
        }));
        let (docker, _) = config_to_docker(
            &cp,
            &mut config,
            Some("langchain/langgraph-api:0.2.47"),
            None,
            None,
            None,
            None,
            false,
        )
        .expect("config_to_docker failed");

        for tool in crate::constants::BUILD_TOOLS {
            let pattern = format!("/usr/local/lib/python*/site-packages/{tool}*");
            assert!(
                docker.contains(&pattern),
                "keep_pkg_tools=false should contain rm of {tool}"
            );
        }
        assert!(
            docker.contains("RUN pip uninstall -y pip setuptools wheel"),
            "keep_pkg_tools=false should uninstall all build tools"
        );
    }

    #[test]
    fn test_config_retain_build_tools_keep_some() {
        // keep_pkg_tools=["pip", "setuptools"] -> uninstall only "wheel", rm only wheel
        let cp = config_path();
        let mut config = make_config(json!({
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./graphs/agent.py:graph"},
            "keep_pkg_tools": ["pip", "setuptools"]
        }));
        let (docker, _) = config_to_docker(
            &cp,
            &mut config,
            Some("langchain/langgraph-api:0.2.47"),
            None,
            None,
            None,
            None,
            false,
        )
        .expect("config_to_docker failed");

        // wheel should be removed
        assert!(
            docker.contains("/usr/local/lib/python*/site-packages/wheel*"),
            "Should contain rm of wheel"
        );
        // pip and setuptools should NOT be removed
        assert!(
            !docker.contains("/usr/local/lib/python*/site-packages/pip*"),
            "Should not contain rm of pip"
        );
        assert!(
            !docker.contains("/usr/local/lib/python*/site-packages/setuptools*"),
            "Should not contain rm of setuptools"
        );
        assert!(
            docker.contains("RUN pip uninstall -y wheel"),
            "Should uninstall only wheel"
        );
        assert!(
            !docker.contains("RUN pip uninstall -y pip setuptools"),
            "Should not uninstall pip or setuptools"
        );
    }

    // -- test_config_to_docker_local_deps --
    #[test]
    fn test_config_to_docker_local_deps() {
        let cp = config_path();
        let mut config = make_config(json!({
            "dependencies": ["./graphs"],
            "graphs": {"agent": "./graphs/agent.py:graph"}
        }));
        let (actual, additional_contexts) = config_to_docker(
            &cp,
            &mut config,
            Some("langchain/langgraph-api-custom"),
            None,
            None,
            None,
            None,
            false,
        )
        .expect("config_to_docker failed");

        let cleaned = clean_empty_lines(&actual);

        // Verify FROM line
        assert!(
            cleaned.starts_with("FROM langchain/langgraph-api-custom:3.11"),
            "Should start with custom base image and python version"
        );

        // Verify faux package for graphs directory
        assert!(
            cleaned.contains("# -- Adding non-package dependency graphs --"),
            "Should add graphs as non-package dependency"
        );
        assert!(
            cleaned.contains("ADD ./graphs /deps/outer-graphs/src"),
            "Should ADD graphs to /deps/outer-graphs/src"
        );
        assert!(
            cleaned.contains("name = \"graphs\""),
            "Should create faux pyproject.toml for graphs"
        );

        // Verify graph path rewritten
        assert!(
            cleaned.contains("/deps/outer-graphs/src/agent.py:graph"),
            "Graph path should be rewritten to container path"
        );

        assert!(additional_contexts.is_empty());
    }

    // -- test_config_to_docker_pipconfig --
    #[test]
    fn test_config_to_docker_pipconfig() {
        let cp = config_path();
        let mut config = make_config(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "pip_config_file": "pipconfig.txt"
        }));
        let (actual, additional_contexts) = config_to_docker(
            &cp,
            &mut config,
            Some("langchain/langgraph-api"),
            None,
            None,
            None,
            None,
            false,
        )
        .expect("config_to_docker failed");

        let cleaned = clean_empty_lines(&actual);

        assert!(
            cleaned.contains("ADD pipconfig.txt /pipconfig.txt"),
            "Should add pipconfig.txt"
        );
        assert!(
            cleaned.contains("PIP_CONFIG_FILE=/pipconfig.txt"),
            "Should use PIP_CONFIG_FILE in install commands"
        );

        assert!(additional_contexts.is_empty());
    }

    // -- test_config_to_docker_end_to_end --
    #[test]
    fn test_config_to_docker_end_to_end() {
        let cp = config_path();
        let mut config = make_config(json!({
            "python_version": "3.12",
            "dependencies": ["./graphs/", "langchain", "langchain_openai"],
            "graphs": {"agent": "./graphs/agent.py:graph"},
            "pip_config_file": "pipconfig.txt",
            "dockerfile_lines": ["ARG meow", "ARG foo"]
        }));
        let (actual, additional_contexts) = config_to_docker(
            &cp,
            &mut config,
            Some("langchain/langgraph-api"),
            None,
            None,
            None,
            None,
            false,
        )
        .expect("config_to_docker failed");

        let cleaned = clean_empty_lines(&actual);

        // FROM line
        assert!(
            cleaned.starts_with("FROM langchain/langgraph-api:3.12"),
            "Should start with FROM line using python 3.12"
        );

        // Dockerfile lines
        assert!(cleaned.contains("ARG meow"), "Should contain dockerfile_lines");
        assert!(cleaned.contains("ARG foo"), "Should contain dockerfile_lines");

        // pip config
        assert!(
            cleaned.contains("ADD pipconfig.txt /pipconfig.txt"),
            "Should add pipconfig"
        );

        // PyPI deps
        assert!(
            cleaned.contains("langchain langchain_openai"),
            "Should install pypi dependencies"
        );

        // Local graphs dep as faux package
        assert!(
            cleaned.contains("# -- Adding non-package dependency graphs --"),
            "Should add graphs as non-package dependency"
        );

        assert!(additional_contexts.is_empty());
    }

    // -- test_config_to_docker_multiplatform --
    #[test]
    fn test_config_to_docker_multiplatform() {
        let cp = config_path();
        let mut config = make_config(json!({
            "node_version": "22",
            "dependencies": ["."],
            "graphs": {
                "python": "./multiplatform/python.py:graph",
                "js": "./multiplatform/js.mts:graph"
            }
        }));
        let (actual, additional_contexts) = config_to_docker(
            &cp,
            &mut config,
            Some("langchain/langgraph-api"),
            None,
            None,
            None,
            None,
            false,
        )
        .expect("config_to_docker failed");

        let cleaned = clean_empty_lines(&actual);

        // Should be Python path since python_version is set
        assert!(
            cleaned.starts_with("FROM langchain/langgraph-api:3.11"),
            "Multiplatform should default to Python base image"
        );

        // Should install Node.js
        assert!(
            cleaned.contains("ENV NODE_VERSION=22"),
            "Should set NODE_VERSION=22"
        );
        assert!(
            cleaned.contains("RUN /storage/install-node.sh"),
            "Should install Node.js"
        );

        // Should have JS dependencies install section
        assert!(
            cleaned.contains("# -- Installing JS dependencies --"),
            "Should have JS dependencies install section"
        );

        assert!(additional_contexts.is_empty());
    }

    // -- test_config_to_docker_gen_ui_python --
    #[test]
    fn test_config_to_docker_gen_ui_python() {
        let cp = config_path();
        let mut config = make_config(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "ui": {"agent": "./graphs/agent.ui.jsx"},
            "ui_config": {"shared": ["nuqs"]}
        }));
        let (actual, additional_contexts) = config_to_docker(
            &cp,
            &mut config,
            Some("langchain/langgraph-api"),
            None,
            None,
            None,
            None,
            false,
        )
        .expect("config_to_docker failed");

        let cleaned = clean_empty_lines(&actual);

        // Should install Node.js for UI
        assert!(
            cleaned.contains("RUN /storage/install-node.sh"),
            "Should install Node.js for UI"
        );

        // Should have UI env vars
        assert!(
            cleaned.contains("ENV LANGGRAPH_UI="),
            "Should set LANGGRAPH_UI"
        );
        assert!(
            cleaned.contains("ENV LANGGRAPH_UI_CONFIG="),
            "Should set LANGGRAPH_UI_CONFIG"
        );

        // Should have JS install section
        assert!(
            cleaned.contains("# -- Installing JS dependencies --"),
            "Should have JS dependencies section for UI"
        );

        assert!(additional_contexts.is_empty());
    }

    // -- test_config_to_docker_webhooks --
    #[test]
    fn test_config_to_docker_webhooks_python() {
        let cp = config_path();
        let webhooks = json!({
            "env_prefix": "LG_WEBHOOK_",
            "url": {
                "require_https": true,
                "allowed_domains": ["hooks.example.com", "*.example.org"],
                "allowed_ports": [443],
                "max_url_length": 1024,
                "disable_loopback": false
            },
            "headers": {
                "x-auth": "${{ env.LG_WEBHOOK_TOKEN }}",
                "x-mixed": "Bearer ${{ env.LG_WEBHOOK_TOKEN }}-suffix"
            }
        });
        let mut config = make_config(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "webhooks": webhooks
        }));
        let (dockerfile, _) = config_to_docker(
            &cp,
            &mut config,
            Some("langchain/langgraph-api"),
            None,
            None,
            None,
            None,
            false,
        )
        .expect("config_to_docker failed");

        assert!(
            dockerfile.contains("ENV LANGGRAPH_WEBHOOKS="),
            "Dockerfile should contain LANGGRAPH_WEBHOOKS env"
        );

        // Extract and parse the JSON from the ENV line
        for line in dockerfile.lines() {
            if line.starts_with("ENV LANGGRAPH_WEBHOOKS='") && line.ends_with('\'') {
                let json_str = &line["ENV LANGGRAPH_WEBHOOKS='".len()..line.len() - 1];
                let parsed: serde_json::Value =
                    serde_json::from_str(json_str).expect("Failed to parse webhook JSON");
                assert_eq!(
                    parsed["env_prefix"], "LG_WEBHOOK_",
                    "Webhook env_prefix should round-trip"
                );
                assert_eq!(
                    parsed["url"]["require_https"], true,
                    "Webhook url.require_https should round-trip"
                );
                return;
            }
        }
        panic!("LANGGRAPH_WEBHOOKS ENV line not found");
    }

    #[test]
    fn test_config_to_docker_webhooks_node() {
        let cp = config_path();
        let webhooks = json!({
            "env_prefix": "LG_WEBHOOK_",
            "url": {"require_https": true},
            "headers": {"x-auth": "${{ env.LG_WEBHOOK_TOKEN }}"}
        });
        let mut config = make_config(json!({
            "node_version": "20",
            "graphs": {"agent": "./graphs/agent.js:graph"},
            "webhooks": webhooks
        }));
        let (dockerfile, _) = config_to_docker(
            &cp,
            &mut config,
            Some("langchain/langgraphjs-api"),
            None,
            None,
            None,
            None,
            false,
        )
        .expect("config_to_docker failed");

        assert!(
            dockerfile.contains("ENV LANGGRAPH_WEBHOOKS="),
            "Node.js Dockerfile should contain LANGGRAPH_WEBHOOKS env"
        );

        for line in dockerfile.lines() {
            if line.starts_with("ENV LANGGRAPH_WEBHOOKS='") && line.ends_with('\'') {
                let json_str = &line["ENV LANGGRAPH_WEBHOOKS='".len()..line.len() - 1];
                let parsed: serde_json::Value =
                    serde_json::from_str(json_str).expect("Failed to parse webhook JSON");
                assert_eq!(parsed["env_prefix"], "LG_WEBHOOK_");
                return;
            }
        }
        panic!("LANGGRAPH_WEBHOOKS ENV line not found in Node.js Dockerfile");
    }

    // -- test_config_to_docker_python_encryption_formatted --
    #[test]
    fn test_config_to_docker_python_encryption_formatted() {
        let cp = config_path();
        let mut config = make_config(json!({
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./graphs/agent.py:graph"},
            "encryption": {"path": "./agent.py:my_encryption"}
        }));
        let (actual, _) = config_to_docker(
            &cp,
            &mut config,
            Some("langchain/langgraph-api"),
            None,
            None,
            None,
            None,
            false,
        )
        .expect("config_to_docker failed");

        assert!(
            actual.contains("LANGGRAPH_ENCRYPTION="),
            "Dockerfile should contain LANGGRAPH_ENCRYPTION"
        );
        assert!(
            actual.contains("/deps/outer-unit_tests/unit_tests/agent.py:my_encryption"),
            "Encryption path should be rewritten to container path"
        );
    }

    // -- test_config_to_compose --
    #[test]
    fn test_config_to_compose_simple() {
        let cp = config_path();
        let mut config = make_config(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"}
        }));
        let actual = config_to_compose(
            &cp,
            &mut config,
            Some("langchain/langgraph-api"),
            None,
            None,
            false,
        )
        .expect("config_to_compose failed");

        let cleaned = clean_empty_lines(&actual);

        assert!(
            cleaned.contains("pull_policy: build"),
            "Compose should contain pull_policy: build"
        );
        assert!(
            cleaned.contains("context: ."),
            "Compose should contain context: ."
        );
        assert!(
            cleaned.contains("dockerfile_inline: |"),
            "Compose should contain dockerfile_inline"
        );
        assert!(
            cleaned.contains("FROM langchain/langgraph-api:3.11"),
            "Compose should contain the FROM line"
        );
        // Escaped $$ for compose
        assert!(
            cleaned.contains("$$dep"),
            "Compose should escape variables with $$"
        );
    }

    #[test]
    fn test_config_to_compose_env_vars() {
        let cp = config_path();
        let mut config = make_config(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "env": {"OPENAI_API_KEY": "key"}
        }));
        let actual = config_to_compose(
            &cp,
            &mut config,
            Some("langchain/langgraph-api-custom"),
            None,
            None,
            false,
        )
        .expect("config_to_compose failed");

        assert!(
            actual.contains("OPENAI_API_KEY: \"key\""),
            "Compose should contain environment variable"
        );
    }

    #[test]
    fn test_config_to_compose_env_file() {
        let cp = config_path();
        let mut config = make_config(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "env": ".env"
        }));
        let actual = config_to_compose(
            &cp,
            &mut config,
            Some("langchain/langgraph-api"),
            None,
            None,
            false,
        )
        .expect("config_to_compose failed");

        assert!(
            actual.contains("env_file: .env"),
            "Compose should contain env_file: .env"
        );
    }

    #[test]
    fn test_config_to_compose_watch() {
        let cp = config_path();
        let mut config = make_config(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"}
        }));
        let actual = config_to_compose(
            &cp,
            &mut config,
            Some("langchain/langgraph-api"),
            None,
            None,
            true,
        )
        .expect("config_to_compose failed");

        let cleaned = clean_empty_lines(&actual);

        assert!(
            cleaned.contains("develop:"),
            "Compose with watch should contain develop:"
        );
        assert!(
            cleaned.contains("watch:"),
            "Compose with watch should contain watch:"
        );
        assert!(
            cleaned.contains("path: test_config.json"),
            "Watch should include config file"
        );
        assert!(
            cleaned.contains("path: ."),
            "Watch should include dependency path"
        );
        assert!(
            cleaned.contains("action: rebuild"),
            "Watch action should be rebuild"
        );
    }

    #[test]
    fn test_config_to_compose_with_api_version() {
        let cp = config_path();
        // Python
        let mut config = make_config(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"}
        }));
        let actual = config_to_compose(
            &cp,
            &mut config,
            Some("langchain/langgraph-api"),
            Some("0.2.74"),
            None,
            false,
        )
        .expect("config_to_compose failed");
        assert!(
            actual.contains("FROM langchain/langgraph-api:0.2.74-py3.11"),
            "Python compose should contain versioned FROM"
        );

        // Node.js
        let mut config = make_config(json!({
            "node_version": "20",
            "graphs": {"agent": "./graphs/agent.js:graph"}
        }));
        let actual = config_to_compose(
            &cp,
            &mut config,
            Some("langchain/langgraphjs-api"),
            Some("0.2.74"),
            None,
            false,
        )
        .expect("config_to_compose failed");
        assert!(
            actual.contains("FROM langchain/langgraphjs-api:0.2.74-node20"),
            "Node.js compose should contain versioned FROM"
        );
    }

    // -- test helpers: image_supports_uv --
    #[test]
    fn test_image_supports_uv() {
        assert!(image_supports_uv("langchain/langgraph-api:0.2.47"));
        assert!(image_supports_uv("langchain/langgraph-api:0.2.48"));
        assert!(image_supports_uv("langchain/langgraph-api:1.0.0"));
        assert!(!image_supports_uv("langchain/langgraph-api:0.2.46"));
        assert!(!image_supports_uv("langchain/langgraph-api:0.2.0"));
        assert!(!image_supports_uv("langchain/langgraph-trial"));
        // No version tag -> supports uv by default
        assert!(image_supports_uv("langchain/langgraph-api"));
    }

    // -- test helpers: get_build_tools_to_uninstall --
    #[test]
    fn test_get_build_tools_to_uninstall() {
        // None -> all tools
        let config = Config::default();
        let tools = get_build_tools_to_uninstall(&config);
        assert_eq!(tools, vec!["pip", "setuptools", "wheel"]);

        // Some(Bool(true)) -> empty
        let config = Config {
            keep_pkg_tools: Some(KeepPkgTools::Bool(true)),
            ..Default::default()
        };
        let tools = get_build_tools_to_uninstall(&config);
        assert!(tools.is_empty());

        // Some(Bool(false)) -> all tools
        let config = Config {
            keep_pkg_tools: Some(KeepPkgTools::Bool(false)),
            ..Default::default()
        };
        let tools = get_build_tools_to_uninstall(&config);
        assert_eq!(tools, vec!["pip", "setuptools", "wheel"]);

        // Some(List) -> only tools not in list
        let config = Config {
            keep_pkg_tools: Some(KeepPkgTools::List(vec![
                "pip".to_string(),
                "setuptools".to_string(),
            ])),
            ..Default::default()
        };
        let tools = get_build_tools_to_uninstall(&config);
        assert_eq!(tools, vec!["wheel"]);
    }
}
