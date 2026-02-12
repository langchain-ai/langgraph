pub mod docker_tag;
pub mod local_deps;
pub mod path_rewrite;
pub mod schema;

use std::path::Path;

use schema::{Config, EnvConfig, GraphSpec, KeepPkgTools};

use crate::constants::{
    BUILD_TOOLS, DEFAULT_IMAGE_DISTRO, DEFAULT_NODE_VERSION, DEFAULT_PYTHON_VERSION,
    MIN_NODE_VERSION, MIN_PYTHON_VERSION, VALID_DISTROS, VALID_PIP_INSTALLERS,
};

/// Check if a graph spec references a Node.js file based on extension.
pub fn is_node_graph(spec: &GraphSpec) -> bool {
    let path_str = match spec {
        GraphSpec::Path(s) => s.as_str(),
        GraphSpec::Dict(m) => match m.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return false,
        },
    };

    let file_path = path_str.split(':').next().unwrap_or("");
    matches!(
        Path::new(file_path)
            .extension()
            .and_then(|e| e.to_str()),
        Some("ts" | "mts" | "cts" | "js" | "mjs" | "cjs")
    )
}

/// Parse a Python version string "major.minor" into (major, minor).
fn parse_version(version_str: &str) -> Result<(u32, u32), String> {
    let cleaned = version_str.split('-').next().unwrap_or(version_str);
    let parts: Vec<&str> = cleaned.split('.').collect();
    if parts.len() != 2 {
        return Err(format!("Invalid version format: {version_str}"));
    }
    let major: u32 = parts[0]
        .parse()
        .map_err(|_| format!("Invalid version format: {version_str}"))?;
    let minor: u32 = parts[1]
        .parse()
        .map_err(|_| format!("Invalid version format: {version_str}"))?;
    Ok((major, minor))
}

/// Parse a Node.js version string (major only) into u32.
fn parse_node_version(version_str: &str) -> Result<u32, String> {
    if version_str.contains('.') {
        return Err(format!(
            "Invalid Node.js version format: {version_str}. Use major version only (e.g., '20')."
        ));
    }
    version_str
        .parse::<u32>()
        .map_err(|_| format!("Invalid Node.js version format: {version_str}. Use major version only (e.g., '20')."))
}

/// Validate a configuration dictionary.
pub fn validate_config(mut config: Config) -> Result<Config, String> {
    let graphs = &config.graphs;

    let some_node = graphs.values().any(|spec| is_node_graph(spec));
    let some_python = graphs.values().any(|spec| !is_node_graph(spec));

    // Set defaults for node_version and python_version
    if config.node_version.is_none() && some_node {
        config.node_version = Some(DEFAULT_NODE_VERSION.to_string());
    }
    if config.python_version.is_none() && some_python {
        config.python_version = Some(DEFAULT_PYTHON_VERSION.to_string());
    }

    // Default image_distro
    if config.image_distro.is_none() {
        config.image_distro = Some(DEFAULT_IMAGE_DISTRO.to_string());
    }

    // Default pip_installer
    if config.pip_installer.is_none() {
        config.pip_installer = Some("auto".to_string());
    }

    // Default env
    if config.env.is_none() {
        config.env = Some(EnvConfig::default());
    }

    // Validate _INTERNAL_docker_tag vs api_version
    if config.internal_docker_tag.is_some() && config.api_version.is_some() {
        return Err("Cannot specify both _INTERNAL_docker_tag and api_version.".to_string());
    }

    // Validate api_version format
    if let Some(ref api_version) = config.api_version {
        let cleaned = api_version.split('-').next().unwrap_or(api_version);
        let parts: Vec<&str> = cleaned.split('.').collect();
        if parts.len() > 3 {
            return Err("Version must be major or major.minor or major.minor.patch.".to_string());
        }
        for part in &parts {
            part.parse::<u32>()
                .map_err(|_| format!("Invalid version format: {api_version}"))?;
        }
    }

    // Validate node_version
    if let Some(ref node_version) = config.node_version {
        let major = parse_node_version(node_version)?;
        if major < MIN_NODE_VERSION {
            return Err(format!(
                "Node.js version {node_version} is not supported. Minimum required version is {MIN_NODE_VERSION}."
            ));
        }
    }

    // Validate python_version
    if let Some(ref pyversion) = config.python_version {
        let cleaned = pyversion.split('-').next().unwrap_or(pyversion);
        if cleaned.split('.').count() != 2
            || !cleaned.split('.').all(|p| p.chars().all(|c| c.is_ascii_digit()))
        {
            return Err(format!(
                "Invalid Python version format: {pyversion}. \
                 Use 'major.minor' format (e.g., '3.11'). \
                 Patch version cannot be specified."
            ));
        }
        if parse_version(pyversion)? < MIN_PYTHON_VERSION {
            return Err(format!(
                "Python version {pyversion} is not supported. \
                 Minimum required version is {}.{}.",
                MIN_PYTHON_VERSION.0, MIN_PYTHON_VERSION.1
            ));
        }

        if config.dependencies.is_empty() {
            return Err(
                "No dependencies found in config. Add at least one dependency to 'dependencies' list."
                    .to_string(),
            );
        }
    }

    // Validate graphs
    if config.graphs.is_empty() {
        return Err(
            "No graphs found in config. Add at least one graph to 'graphs' dictionary."
                .to_string(),
        );
    }

    // Validate image_distro
    if let Some(ref distro) = config.image_distro {
        if distro == "bullseye" {
            return Err(
                "Bullseye images were deprecated in version 0.4.13. \
                 Please use 'bookworm' or 'debian' instead."
                    .to_string(),
            );
        }
        if !VALID_DISTROS.contains(&distro.as_str()) {
            return Err(format!(
                "Invalid image_distro: '{distro}'. Must be one of 'debian', 'wolfi', or 'bookworm'."
            ));
        }
    }

    // Validate pip_installer
    if let Some(ref pip_installer) = config.pip_installer {
        if !VALID_PIP_INSTALLERS.contains(&pip_installer.as_str()) {
            return Err(format!(
                "Invalid pip_installer: '{pip_installer}'. Must be 'auto', 'pip', or 'uv'."
            ));
        }
    }

    // Validate auth config
    if let Some(ref auth_conf) = config.auth {
        if let Some(ref path) = auth_conf.path {
            if !path.contains(':') {
                return Err(format!(
                    "Invalid auth.path format: '{path}'. \
                     Must be in format './path/to/file.py:attribute_name'"
                ));
            }
        }
    }

    // Validate encryption config
    if let Some(ref encryption_conf) = config.encryption {
        if let Some(ref path) = encryption_conf.path {
            if !path.contains(':') {
                return Err(format!(
                    "Invalid encryption.path format: '{path}'. \
                     Must be in format './path/to/file.py:attribute_name'"
                ));
            }
        }
    }

    // Validate http config
    if let Some(ref http_conf) = config.http {
        if let Some(ref app) = http_conf.app {
            if !app.contains(':') {
                return Err(format!(
                    "Invalid http.app format: '{app}'. \
                     Must be in format './path/to/file.py:attribute_name'"
                ));
            }
        }
    }

    // Validate keep_pkg_tools
    if let Some(ref keep_pkg_tools) = config.keep_pkg_tools {
        match keep_pkg_tools {
            KeepPkgTools::List(tools) => {
                for tool in tools {
                    if !BUILD_TOOLS.contains(&tool.as_str()) {
                        return Err(format!(
                            "Invalid keep_pkg_tools: '{tool}'. \
                             Must be one of 'pip', 'setuptools', 'wheel'."
                        ));
                    }
                }
            }
            KeepPkgTools::Bool(_) => {}
        }
    }

    Ok(config)
}

/// Load and validate a configuration file.
pub fn validate_config_file(config_path: &Path) -> Result<Config, String> {
    let content = std::fs::read_to_string(config_path)
        .map_err(|e| format!("Could not read config file {}: {e}", config_path.display()))?;

    let config: Config = serde_json::from_str(&content)
        .map_err(|e| format!("Invalid JSON in config file {}: {e}", config_path.display()))?;

    let validated = validate_config(config)?;

    // Check package.json engines
    if validated.node_version.is_some() {
        let package_json_path = config_path.parent().unwrap().join("package.json");
        if package_json_path.is_file() {
            let pkg_content = std::fs::read_to_string(&package_json_path)
                .map_err(|e| format!("Could not read package.json: {e}"))?;
            let pkg: serde_json::Value = serde_json::from_str(&pkg_content).map_err(|_| {
                format!(
                    "Invalid package.json found in langgraph config directory {}: file is not valid JSON",
                    package_json_path.display()
                )
            })?;

            if let Some(engines) = pkg.get("engines").and_then(|e| e.as_object()) {
                // Only 'node' engine is supported
                if engines.keys().any(|k| k != "node") {
                    return Err(format!(
                        "Only 'node' engine is supported in package.json engines. Got engines: {:?}",
                        engines.keys().collect::<Vec<_>>()
                    ));
                }
                if let Some(node_version) = engines.get("node").and_then(|v| v.as_str()) {
                    let major = parse_node_version(node_version)?;
                    if major < MIN_NODE_VERSION {
                        return Err(format!(
                            "Node.js version in package.json engines must be >= {MIN_NODE_VERSION} \
                             (major version only), got '{node_version}'. Minor/patch versions \
                             (like '20.x.y') are not supported to prevent deployment issues \
                             when new Node.js versions are released."
                        ));
                    }
                }
            }
        }
    }

    Ok(validated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;

    fn config_from_json(val: serde_json::Value) -> Config {
        serde_json::from_value::<Config>(val).unwrap()
    }

    // ---------------------------------------------------------------
    // test_validate_config
    // ---------------------------------------------------------------

    #[test]
    fn test_validate_config_minimal() {
        let config = config_from_json(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"}
        }));
        let result = validate_config(config).unwrap();
        assert_eq!(result.python_version.as_deref(), Some("3.11"));
        assert_eq!(result.node_version, None);
        assert_eq!(result.pip_installer.as_deref(), Some("auto"));
        assert_eq!(result.image_distro.as_deref(), Some("debian"));
        assert_eq!(result.base_image, None);
    }

    #[test]
    fn test_validate_config_full() {
        let config = config_from_json(json!({
            "python_version": "3.12",
            "dependencies": [".", "langchain"],
            "graphs": {"agent": "./agent.py:graph"},
            "pip_config_file": "/etc/pip.conf",
            "dockerfile_lines": ["RUN apt-get update"],
            "env": ".env"
        }));
        let result = validate_config(config).unwrap();
        assert_eq!(result.python_version.as_deref(), Some("3.12"));
        assert_eq!(result.pip_config_file.as_deref(), Some("/etc/pip.conf"));
        assert_eq!(result.dockerfile_lines, vec!["RUN apt-get update"]);
        assert!(matches!(result.env, Some(EnvConfig::File(ref s)) if s == ".env"));
    }

    #[test]
    fn test_validate_config_python_313() {
        let config = config_from_json(json!({
            "python_version": "3.13",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"}
        }));
        let result = validate_config(config).unwrap();
        assert_eq!(result.python_version.as_deref(), Some("3.13"));
    }

    #[test]
    fn test_validate_config_python_39_error() {
        let config = config_from_json(json!({
            "python_version": "3.9",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"}
        }));
        let err = validate_config(config).unwrap_err();
        assert!(err.contains("Minimum required version"), "Expected 'Minimum required version' but got: {err}");
    }

    #[test]
    fn test_validate_config_missing_dependencies() {
        let config = config_from_json(json!({
            "python_version": "3.11",
            "graphs": {"agent": "./agent.py:graph"}
        }));
        let err = validate_config(config).unwrap_err();
        assert!(err.contains("dependencies"), "Expected error about dependencies but got: {err}");
    }

    #[test]
    fn test_validate_config_missing_graphs() {
        let config = config_from_json(json!({
            "dependencies": ["."]
        }));
        let err = validate_config(config).unwrap_err();
        assert!(err.contains("graphs"), "Expected error about graphs but got: {err}");
    }

    #[test]
    fn test_validate_config_python_version_with_patch() {
        let config = config_from_json(json!({
            "python_version": "3.11.0",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"}
        }));
        let err = validate_config(config).unwrap_err();
        assert!(err.contains("Invalid Python version format"), "Expected 'Invalid Python version format' but got: {err}");
    }

    #[test]
    fn test_validate_config_python_version_major_only() {
        let config = config_from_json(json!({
            "python_version": "3",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"}
        }));
        let err = validate_config(config).unwrap_err();
        assert!(err.contains("Invalid Python version format"), "Expected 'Invalid Python version format' but got: {err}");
    }

    #[test]
    fn test_validate_config_python_version_non_numeric() {
        let config = config_from_json(json!({
            "python_version": "abc.def",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"}
        }));
        let err = validate_config(config).unwrap_err();
        assert!(err.contains("Invalid Python version format"), "Expected 'Invalid Python version format' but got: {err}");
    }

    #[test]
    fn test_validate_config_python_310_error() {
        let config = config_from_json(json!({
            "python_version": "3.10",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"}
        }));
        let err = validate_config(config).unwrap_err();
        assert!(err.contains("Minimum required version"), "Expected 'Minimum required version' but got: {err}");
    }

    #[test]
    fn test_validate_config_python_312_slim() {
        let config = config_from_json(json!({
            "python_version": "3.12-slim",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"}
        }));
        let result = validate_config(config).unwrap();
        assert_eq!(result.python_version.as_deref(), Some("3.12-slim"));
    }

    #[test]
    fn test_validate_config_http_app_no_colon() {
        let config = config_from_json(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "http": {"app": "../../examples/my_app.py"}
        }));
        let err = validate_config(config).unwrap_err();
        assert!(err.contains("Invalid http.app format"), "Expected 'Invalid http.app format' but got: {err}");
    }

    // ---------------------------------------------------------------
    // test_validate_config_image_distro
    // ---------------------------------------------------------------

    #[test]
    fn test_validate_config_image_distro_debian() {
        let config = config_from_json(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "image_distro": "debian"
        }));
        let result = validate_config(config).unwrap();
        assert_eq!(result.image_distro.as_deref(), Some("debian"));
    }

    #[test]
    fn test_validate_config_image_distro_wolfi() {
        let config = config_from_json(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "image_distro": "wolfi"
        }));
        let result = validate_config(config).unwrap();
        assert_eq!(result.image_distro.as_deref(), Some("wolfi"));
    }

    #[test]
    fn test_validate_config_image_distro_default() {
        let config = config_from_json(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"}
        }));
        let result = validate_config(config).unwrap();
        assert_eq!(result.image_distro.as_deref(), Some("debian"));
    }

    #[test]
    fn test_validate_config_image_distro_bullseye_error() {
        let config = config_from_json(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "image_distro": "bullseye"
        }));
        let err = validate_config(config).unwrap_err();
        assert!(err.contains("Bullseye images were deprecated"), "Expected 'Bullseye images were deprecated' but got: {err}");
    }

    #[test]
    fn test_validate_config_image_distro_ubuntu_error() {
        let config = config_from_json(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "image_distro": "ubuntu"
        }));
        let err = validate_config(config).unwrap_err();
        assert!(err.contains("Invalid image_distro: 'ubuntu'"), "Expected \"Invalid image_distro: 'ubuntu'\" but got: {err}");
    }

    #[test]
    fn test_validate_config_image_distro_alpine_error() {
        let config = config_from_json(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "image_distro": "alpine"
        }));
        let err = validate_config(config).unwrap_err();
        assert!(err.contains("Invalid image_distro: 'alpine'"), "Expected \"Invalid image_distro: 'alpine'\" but got: {err}");
    }

    #[test]
    fn test_validate_config_node_with_wolfi() {
        let config = config_from_json(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.mts:graph"},
            "image_distro": "wolfi"
        }));
        let result = validate_config(config).unwrap();
        assert_eq!(result.image_distro.as_deref(), Some("wolfi"));
        assert_eq!(result.node_version.as_deref(), Some("20"));
    }

    #[test]
    fn test_validate_config_node_default_distro() {
        let config = config_from_json(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.mts:graph"}
        }));
        let result = validate_config(config).unwrap();
        assert_eq!(result.image_distro.as_deref(), Some("debian"));
    }

    // ---------------------------------------------------------------
    // test_validate_config_pip_installer
    // ---------------------------------------------------------------

    #[test]
    fn test_validate_config_pip_installer_auto() {
        let config = config_from_json(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "pip_installer": "auto"
        }));
        let result = validate_config(config).unwrap();
        assert_eq!(result.pip_installer.as_deref(), Some("auto"));
    }

    #[test]
    fn test_validate_config_pip_installer_pip() {
        let config = config_from_json(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "pip_installer": "pip"
        }));
        let result = validate_config(config).unwrap();
        assert_eq!(result.pip_installer.as_deref(), Some("pip"));
    }

    #[test]
    fn test_validate_config_pip_installer_uv() {
        let config = config_from_json(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "pip_installer": "uv"
        }));
        let result = validate_config(config).unwrap();
        assert_eq!(result.pip_installer.as_deref(), Some("uv"));
    }

    #[test]
    fn test_validate_config_pip_installer_default() {
        let config = config_from_json(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"}
        }));
        let result = validate_config(config).unwrap();
        assert_eq!(result.pip_installer.as_deref(), Some("auto"));
    }

    #[test]
    fn test_validate_config_pip_installer_conda_error() {
        let config = config_from_json(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "pip_installer": "conda"
        }));
        let err = validate_config(config).unwrap_err();
        assert!(err.contains("Invalid pip_installer: 'conda'"), "Expected \"Invalid pip_installer: 'conda'\" but got: {err}");
        assert!(err.contains("Must be 'auto', 'pip', or 'uv'"), "Expected \"Must be 'auto', 'pip', or 'uv'\" but got: {err}");
    }

    #[test]
    fn test_validate_config_pip_installer_invalid_error() {
        let config = config_from_json(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "pip_installer": "invalid"
        }));
        let err = validate_config(config).unwrap_err();
        assert!(err.contains("Invalid pip_installer: 'invalid'"), "Expected \"Invalid pip_installer: 'invalid'\" but got: {err}");
    }

    // ---------------------------------------------------------------
    // test_validate_config_multiplatform
    // ---------------------------------------------------------------

    #[test]
    fn test_validate_config_js_only_no_explicit_versions() {
        let config = config_from_json(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./js.mts:graph"}
        }));
        let result = validate_config(config).unwrap();
        assert_eq!(result.node_version.as_deref(), Some("20"));
        assert_eq!(result.python_version, None);
    }

    #[test]
    fn test_validate_config_both_versions_explicit() {
        let config = config_from_json(json!({
            "python_version": "3.12",
            "node_version": "22",
            "dependencies": ["."],
            "graphs": {
                "agent_py": "./agent.py:graph",
                "agent_js": "./agent.mts:graph"
            }
        }));
        let result = validate_config(config).unwrap();
        assert_eq!(result.python_version.as_deref(), Some("3.12"));
        assert_eq!(result.node_version.as_deref(), Some("22"));
    }

    #[test]
    fn test_validate_config_mixed_graphs_no_explicit_versions() {
        let config = config_from_json(json!({
            "dependencies": ["."],
            "graphs": {
                "agent_py": "./agent.py:graph",
                "agent_js": "./agent.mts:graph"
            }
        }));
        let result = validate_config(config).unwrap();
        assert_eq!(result.node_version.as_deref(), Some("20"));
        assert_eq!(result.python_version.as_deref(), Some("3.11"));
    }

    #[test]
    fn test_validate_config_mixed_graphs_node_version_only() {
        let config = config_from_json(json!({
            "node_version": "22",
            "dependencies": ["."],
            "graphs": {
                "agent_py": "./agent.py:graph",
                "agent_js": "./agent.mts:graph"
            }
        }));
        let result = validate_config(config).unwrap();
        assert_eq!(result.node_version.as_deref(), Some("22"));
        assert_eq!(result.python_version.as_deref(), Some("3.11"));
    }

    #[test]
    fn test_validate_config_mixed_graphs_python_version_only() {
        let config = config_from_json(json!({
            "python_version": "3.12",
            "dependencies": ["."],
            "graphs": {
                "agent_py": "./agent.py:graph",
                "agent_js": "./agent.mts:graph"
            }
        }));
        let result = validate_config(config).unwrap();
        assert_eq!(result.node_version.as_deref(), Some("20"));
        assert_eq!(result.python_version.as_deref(), Some("3.12"));
    }

    #[test]
    fn test_validate_config_unknown_extension_assumes_python() {
        let config = config_from_json(json!({
            "dependencies": ["."],
            "graphs": {"agent": "local.workflow:graph"}
        }));
        let result = validate_config(config).unwrap();
        assert_eq!(result.python_version.as_deref(), Some("3.11"));
        assert_eq!(result.node_version, None);
    }

    // ---------------------------------------------------------------
    // test_validate_config_encryption
    // ---------------------------------------------------------------

    #[test]
    fn test_validate_config_encryption_valid() {
        let config = config_from_json(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "encryption": {"path": "./encryption.py:encryption"}
        }));
        let result = validate_config(config).unwrap();
        assert!(result.encryption.is_some());
        assert_eq!(
            result.encryption.as_ref().unwrap().path.as_deref(),
            Some("./encryption.py:encryption")
        );
    }

    #[test]
    fn test_validate_config_encryption_no_colon_error() {
        let config = config_from_json(json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "encryption": {"path": "./encryption.py"}
        }));
        let err = validate_config(config).unwrap_err();
        assert!(err.contains("Invalid encryption.path format"), "Expected 'Invalid encryption.path format' but got: {err}");
    }

    // ---------------------------------------------------------------
    // test_validate_config_file
    // ---------------------------------------------------------------

    #[test]
    fn test_validate_config_file_node_config() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("langgraph.json");
        let config_json = json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.mts:graph"}
        });
        let mut f = std::fs::File::create(&config_path).unwrap();
        write!(f, "{}", serde_json::to_string(&config_json).unwrap()).unwrap();
        drop(f);

        let result = validate_config_file(&config_path).unwrap();
        assert_eq!(result.node_version.as_deref(), Some("20"));
    }

    #[test]
    fn test_validate_config_file_with_valid_package_json() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("langgraph.json");
        let config_json = json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.mts:graph"}
        });
        std::fs::write(&config_path, serde_json::to_string(&config_json).unwrap()).unwrap();

        let pkg_path = tmp.path().join("package.json");
        let pkg_json = json!({
            "engines": {"node": "20"}
        });
        std::fs::write(&pkg_path, serde_json::to_string(&pkg_json).unwrap()).unwrap();

        let result = validate_config_file(&config_path).unwrap();
        assert_eq!(result.node_version.as_deref(), Some("20"));
    }

    #[test]
    fn test_validate_config_file_package_json_minor_version_error() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("langgraph.json");
        let config_json = json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.mts:graph"}
        });
        std::fs::write(&config_path, serde_json::to_string(&config_json).unwrap()).unwrap();

        let pkg_path = tmp.path().join("package.json");
        let pkg_json = json!({
            "engines": {"node": "20.18"}
        });
        std::fs::write(&pkg_path, serde_json::to_string(&pkg_json).unwrap()).unwrap();

        let err = validate_config_file(&config_path).unwrap_err();
        assert!(err.contains("Use major version only"), "Expected 'Use major version only' but got: {err}");
    }

    #[test]
    fn test_validate_config_file_package_json_old_node_error() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("langgraph.json");
        let config_json = json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.mts:graph"}
        });
        std::fs::write(&config_path, serde_json::to_string(&config_json).unwrap()).unwrap();

        let pkg_path = tmp.path().join("package.json");
        let pkg_json = json!({
            "engines": {"node": "18"}
        });
        std::fs::write(&pkg_path, serde_json::to_string(&pkg_json).unwrap()).unwrap();

        let err = validate_config_file(&config_path).unwrap_err();
        assert!(err.contains("must be >= 20"), "Expected 'must be >= 20' but got: {err}");
    }

    #[test]
    fn test_validate_config_file_package_json_deno_engine_error() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("langgraph.json");
        let config_json = json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.mts:graph"}
        });
        std::fs::write(&config_path, serde_json::to_string(&config_json).unwrap()).unwrap();

        let pkg_path = tmp.path().join("package.json");
        let pkg_json = json!({
            "engines": {"node": "20", "deno": "1.0"}
        });
        std::fs::write(&pkg_path, serde_json::to_string(&pkg_json).unwrap()).unwrap();

        let err = validate_config_file(&config_path).unwrap_err();
        assert!(err.contains("Only 'node' engine is supported"), "Expected 'Only node engine is supported' but got: {err}");
    }

    #[test]
    fn test_validate_config_file_invalid_package_json() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("langgraph.json");
        let config_json = json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.mts:graph"}
        });
        std::fs::write(&config_path, serde_json::to_string(&config_json).unwrap()).unwrap();

        let pkg_path = tmp.path().join("package.json");
        std::fs::write(&pkg_path, "this is not valid json!!!").unwrap();

        let err = validate_config_file(&config_path).unwrap_err();
        assert!(err.contains("Invalid package.json"), "Expected 'Invalid package.json' but got: {err}");
    }

    #[test]
    fn test_validate_config_file_python_ignores_bad_package_json() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("langgraph.json");
        let config_json = json!({
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"}
        });
        std::fs::write(&config_path, serde_json::to_string(&config_json).unwrap()).unwrap();

        // Write a bad package.json - should be ignored for Python-only config
        let pkg_path = tmp.path().join("package.json");
        std::fs::write(&pkg_path, "this is not valid json!!!").unwrap();

        let result = validate_config_file(&config_path).unwrap();
        assert_eq!(result.python_version.as_deref(), Some("3.11"));
    }
}
