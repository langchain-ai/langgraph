use super::schema::Config;
use crate::constants::DEFAULT_IMAGE_DISTRO;

/// Get the default base image for a config.
pub fn default_base_image(config: &Config) -> String {
    if let Some(ref base) = config.base_image {
        return base.clone();
    }
    if config.node_version.is_some() && config.python_version.is_none() {
        "langchain/langgraphjs-api".to_string()
    } else {
        "langchain/langgraph-api".to_string()
    }
}

/// Build the Docker image tag string.
pub fn docker_tag(
    config: &Config,
    base_image: Option<&str>,
    api_version: Option<&str>,
) -> String {
    let api_version = api_version
        .map(|s| s.to_string())
        .or_else(|| config.api_version.clone());
    let base_image = base_image
        .map(|s| s.to_string())
        .unwrap_or_else(|| default_base_image(config));

    let image_distro = config
        .image_distro
        .as_deref()
        .unwrap_or(DEFAULT_IMAGE_DISTRO);
    let distro_tag = if image_distro == DEFAULT_IMAGE_DISTRO {
        String::new()
    } else {
        format!("-{image_distro}")
    };

    if let Some(ref tag) = config.internal_docker_tag {
        return format!("{base_image}:{tag}");
    }

    // Build the standard tag format
    let (language, version) = if config.node_version.is_some() && config.python_version.is_none() {
        ("node", config.node_version.as_deref().unwrap_or("20"))
    } else {
        (
            "py",
            config
                .python_version
                .as_deref()
                .unwrap_or(crate::constants::DEFAULT_PYTHON_VERSION),
        )
    };

    let version_distro_tag = format!("{version}{distro_tag}");

    if let Some(api_ver) = api_version {
        format!("{base_image}:{api_ver}-{language}{version_distro_tag}")
    } else if base_image.contains("/langgraph-server") && !base_image.contains(&version_distro_tag)
    {
        format!("{base_image}-{language}{version_distro_tag}")
    } else {
        format!("{base_image}:{version_distro_tag}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::Config;
    use crate::config::validate_config;
    use serde_json::json;

    fn config_from_json(v: serde_json::Value) -> Config {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn test_default_base_image_python() {
        let config = Config {
            python_version: Some("3.11".to_string()),
            ..Default::default()
        };
        assert_eq!(default_base_image(&config), "langchain/langgraph-api");
    }

    #[test]
    fn test_default_base_image_node() {
        let config = Config {
            node_version: Some("20".to_string()),
            ..Default::default()
        };
        assert_eq!(default_base_image(&config), "langchain/langgraphjs-api");
    }

    #[test]
    fn test_docker_tag_basic() {
        let config = Config {
            python_version: Some("3.11".to_string()),
            ..Default::default()
        };
        assert_eq!(
            docker_tag(&config, None, None),
            "langchain/langgraph-api:3.11"
        );
    }

    #[test]
    fn test_docker_tag_with_api_version_basic() {
        let config = Config {
            python_version: Some("3.11".to_string()),
            ..Default::default()
        };
        assert_eq!(
            docker_tag(&config, None, Some("0.2.74")),
            "langchain/langgraph-api:0.2.74-py3.11"
        );
    }

    #[test]
    fn test_docker_tag_wolfi() {
        let config = Config {
            python_version: Some("3.12".to_string()),
            image_distro: Some("wolfi".to_string()),
            ..Default::default()
        };
        assert_eq!(
            docker_tag(&config, None, None),
            "langchain/langgraph-api:3.12-wolfi"
        );
    }

    #[test]
    fn test_docker_tag_internal() {
        let config = Config {
            python_version: Some("3.11".to_string()),
            internal_docker_tag: Some("custom-tag".to_string()),
            ..Default::default()
        };
        assert_eq!(
            docker_tag(&config, None, None),
            "langchain/langgraph-api:custom-tag"
        );
    }

    // ---------------------------------------------------------------
    // Comprehensive tests ported from Python test_config.py
    // ---------------------------------------------------------------

    #[test]
    fn test_docker_tag_image_distro() {
        // Test 1: Default distro (debian) - no suffix
        let cfg = config_from_json(json!({
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"}
        }));
        let cfg = validate_config(cfg).unwrap();
        assert_eq!(docker_tag(&cfg, None, None), "langchain/langgraph-api:3.11");

        // Test 2: Explicit debian distro - same as default
        let cfg = config_from_json(json!({
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "image_distro": "debian"
        }));
        let cfg = validate_config(cfg).unwrap();
        assert_eq!(docker_tag(&cfg, None, None), "langchain/langgraph-api:3.11");

        // Test 3: Wolfi distro with python
        let cfg = config_from_json(json!({
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "image_distro": "wolfi"
        }));
        let cfg = validate_config(cfg).unwrap();
        assert_eq!(
            docker_tag(&cfg, None, None),
            "langchain/langgraph-api:3.11-wolfi"
        );

        // Test 4: Node.js with default distro
        let cfg = config_from_json(json!({
            "node_version": "20",
            "graphs": {"agent": "./agent.js:graph"}
        }));
        let cfg = validate_config(cfg).unwrap();
        assert_eq!(
            docker_tag(&cfg, None, None),
            "langchain/langgraphjs-api:20"
        );

        // Test 5: Node.js with wolfi distro
        let cfg = config_from_json(json!({
            "node_version": "20",
            "graphs": {"agent": "./agent.js:graph"},
            "image_distro": "wolfi"
        }));
        let cfg = validate_config(cfg).unwrap();
        assert_eq!(
            docker_tag(&cfg, None, None),
            "langchain/langgraphjs-api:20-wolfi"
        );

        // Test 6: Custom base image with wolfi
        let cfg = config_from_json(json!({
            "python_version": "3.12",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "image_distro": "wolfi",
            "base_image": "my-registry/custom-image"
        }));
        let cfg = validate_config(cfg).unwrap();
        assert_eq!(
            docker_tag(&cfg, Some("my-registry/custom-image"), None),
            "my-registry/custom-image:3.12-wolfi"
        );
    }

    #[test]
    fn test_docker_tag_multiplatform_with_distro() {
        // Test 1: Python + Node with wolfi -> defaults to Python
        let cfg = config_from_json(json!({
            "python_version": "3.11",
            "node_version": "20",
            "dependencies": ["."],
            "graphs": {"python": "./agent.py:graph", "js": "./agent.js:graph"},
            "image_distro": "wolfi"
        }));
        let cfg = validate_config(cfg).unwrap();
        assert_eq!(
            docker_tag(&cfg, None, None),
            "langchain/langgraph-api:3.11-wolfi"
        );

        // Test 2: Node-only with wolfi
        let cfg = config_from_json(json!({
            "node_version": "20",
            "graphs": {"js": "./agent.js:graph"},
            "image_distro": "wolfi"
        }));
        let cfg = validate_config(cfg).unwrap();
        assert_eq!(
            docker_tag(&cfg, None, None),
            "langchain/langgraphjs-api:20-wolfi"
        );
    }

    #[test]
    fn test_docker_tag_different_python_versions_with_distro() {
        for (ver, expected) in &[
            ("3.11", "langchain/langgraph-api:3.11-wolfi"),
            ("3.12", "langchain/langgraph-api:3.12-wolfi"),
            ("3.13", "langchain/langgraph-api:3.13-wolfi"),
        ] {
            let cfg = config_from_json(json!({
                "python_version": ver,
                "dependencies": ["."],
                "graphs": {"agent": "./agent.py:graph"},
                "image_distro": "wolfi"
            }));
            let cfg = validate_config(cfg).unwrap();
            assert_eq!(
                docker_tag(&cfg, None, None),
                *expected,
                "Failed for Python {ver}"
            );
        }
    }

    #[test]
    fn test_docker_tag_different_node_versions_with_distro() {
        for (ver, expected) in &[
            ("20", "langchain/langgraphjs-api:20-wolfi"),
            ("21", "langchain/langgraphjs-api:21-wolfi"),
            ("22", "langchain/langgraphjs-api:22-wolfi"),
        ] {
            let cfg = config_from_json(json!({
                "node_version": ver,
                "graphs": {"agent": "./agent.js:graph"},
                "image_distro": "wolfi"
            }));
            let cfg = validate_config(cfg).unwrap();
            assert_eq!(
                docker_tag(&cfg, None, None),
                *expected,
                "Failed for Node.js {ver}"
            );
        }
    }

    /// Helper to run a single api_version test scenario with both in_config and as-argument modes.
    fn run_api_version_test(
        base_json: serde_json::Value,
        api_version: &str,
        base_image_arg: Option<&str>,
        expected: &str,
    ) {
        // Mode 1: api_version passed as function argument (not in config)
        {
            let cfg = config_from_json(base_json.clone());
            let cfg = validate_config(cfg).unwrap();
            assert_eq!(
                docker_tag(&cfg, base_image_arg, Some(api_version)),
                expected,
                "Failed with api_version as argument"
            );
        }

        // Mode 2: api_version set in config (not passed as argument)
        {
            let mut json_val = base_json.clone();
            json_val
                .as_object_mut()
                .unwrap()
                .insert("api_version".to_string(), json!(api_version));
            let cfg = config_from_json(json_val);
            let cfg = validate_config(cfg).unwrap();
            assert_eq!(
                docker_tag(&cfg, base_image_arg, None),
                expected,
                "Failed with api_version in config"
            );
        }
    }

    #[test]
    fn test_docker_tag_with_api_version() {
        let version = "0.2.74";

        // Test 1: Python config with api_version and default distro
        run_api_version_test(
            json!({
                "python_version": "3.11",
                "dependencies": ["."],
                "graphs": {"agent": "./agent.py:graph"}
            }),
            version,
            None,
            "langchain/langgraph-api:0.2.74-py3.11",
        );

        // Test 2: Python config with api_version and wolfi distro
        run_api_version_test(
            json!({
                "python_version": "3.12",
                "dependencies": ["."],
                "graphs": {"agent": "./agent.py:graph"},
                "image_distro": "wolfi"
            }),
            version,
            None,
            "langchain/langgraph-api:0.2.74-py3.12-wolfi",
        );

        // Test 3: Node.js config with api_version and default distro
        run_api_version_test(
            json!({
                "node_version": "20",
                "graphs": {"agent": "./agent.js:graph"}
            }),
            version,
            None,
            "langchain/langgraphjs-api:0.2.74-node20",
        );

        // Test 4: Node.js config with api_version and wolfi distro
        run_api_version_test(
            json!({
                "node_version": "20",
                "graphs": {"agent": "./agent.js:graph"},
                "image_distro": "wolfi"
            }),
            version,
            None,
            "langchain/langgraphjs-api:0.2.74-node20-wolfi",
        );

        // Test 5: Custom base image with api_version
        run_api_version_test(
            json!({
                "python_version": "3.11",
                "dependencies": ["."],
                "graphs": {"agent": "./agent.py:graph"},
                "base_image": "my-registry/custom-image"
            }),
            version,
            Some("my-registry/custom-image"),
            "my-registry/custom-image:0.2.74-py3.11",
        );

        // Test 6: Different Python versions with api_version
        for py_ver in &["3.11", "3.12", "3.13"] {
            let expected = format!("langchain/langgraph-api:{version}-py{py_ver}");
            run_api_version_test(
                json!({
                    "python_version": py_ver,
                    "dependencies": ["."],
                    "graphs": {"agent": "./agent.py:graph"}
                }),
                version,
                None,
                &expected,
            );
        }

        // Test 7: Without api_version should work as before
        let cfg = config_from_json(json!({
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"}
        }));
        let cfg = validate_config(cfg).unwrap();
        assert_eq!(docker_tag(&cfg, None, None), "langchain/langgraph-api:3.11");

        // Test 8: Multiplatform with api_version (should default to Python)
        run_api_version_test(
            json!({
                "python_version": "3.11",
                "node_version": "20",
                "dependencies": ["."],
                "graphs": {"python": "./agent.py:graph", "js": "./agent.js:graph"}
            }),
            version,
            None,
            "langchain/langgraph-api:0.2.74-py3.11",
        );

        // Test 9: _INTERNAL_docker_tag ignores api_version
        // (can only test with api_version as argument since both in config is invalid)
        let cfg = config_from_json(json!({
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./agent.py:graph"},
            "_INTERNAL_docker_tag": "internal-tag"
        }));
        let cfg = validate_config(cfg).unwrap();
        assert_eq!(
            docker_tag(&cfg, None, Some("0.2.74")),
            "langchain/langgraph-api:internal-tag"
        );

        // Test 10: langgraph-server base image with api_version
        run_api_version_test(
            json!({
                "python_version": "3.11",
                "dependencies": ["."],
                "graphs": {"agent": "./agent.py:graph"}
            }),
            version,
            Some("langchain/langgraph-server"),
            "langchain/langgraph-server:0.2.74-py3.11",
        );
    }
}
