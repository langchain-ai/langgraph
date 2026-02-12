use std::collections::HashMap;
use std::path::Path;

use console::style;

use crate::analytics;
use crate::config::docker_tag::docker_tag;
use crate::config::validate_config_file;
use crate::docker::dockerfile::config_to_docker;
use crate::exec::{run_command, run_command_streaming};
use crate::progress::Progress;
use crate::util::warn_non_wolfi_distro;

/// Build a LangGraph API server Docker image.
#[allow(clippy::too_many_arguments)]
pub fn run(
    config: &str,
    tag: &str,
    pull: bool,
    base_image: Option<&str>,
    api_version: Option<&str>,
    install_command: Option<&str>,
    build_command: Option<&str>,
    docker_build_args: &[String],
) -> Result<(), String> {
    // Fire-and-forget analytics
    let mut params = HashMap::new();
    params.insert("pull".to_string(), pull.to_string());
    analytics::log_command("build", &params);

    // Check docker is available
    if which::which("docker").is_err() {
        return Err("Docker not installed".to_string());
    }

    let progress = Progress::new("Pulling...");

    // Validate config
    let config_path = Path::new(config);
    let mut config_json = validate_config_file(config_path)?;
    let config_value = serde_json::to_value(&config_json).unwrap();
    warn_non_wolfi_distro(&config_value);

    // Pull latest images if requested
    if pull {
        let image_tag = docker_tag(&config_json, base_image, api_version);
        run_command("docker", &["pull", &image_tag], None, true)?;
    }

    progress.set_message("Building...");

    // Determine build context
    let is_js_project =
        config_json.node_version.is_some() && config_json.python_version.is_none();

    // For JS projects with install/build commands, use CWD; otherwise use config parent
    let build_context = if is_js_project && (build_command.is_some() || install_command.is_some()) {
        std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| ".".to_string())
    } else {
        config_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_string_lossy()
            .to_string()
    };

    // Generate Dockerfile
    let (dockerfile_content, additional_contexts) = config_to_docker(
        config_path,
        &mut config_json,
        base_image,
        api_version,
        install_command,
        build_command,
        Some(&build_context),
        false, // no variable escaping for docker build
    )?;

    // Build docker build args
    let mut args: Vec<String> = vec![
        "build".to_string(),
        "-f".to_string(),
        "-".to_string(), // Dockerfile from stdin
        "-t".to_string(),
        tag.to_string(),
    ];

    // Add additional build contexts
    for (name, path) in &additional_contexts {
        args.push("--build-context".to_string());
        args.push(format!("{name}={path}"));
    }

    // Add passthrough docker build args
    for extra in docker_build_args {
        args.push(extra.clone());
    }

    // Add build context as last arg
    args.push(build_context);

    progress.finish();

    // Run docker build with streaming output
    let args_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    run_command_streaming("docker", &args_refs, Some(&dockerfile_content), true)?;

    eprintln!(
        "{}",
        style(format!("Successfully built image: {tag}")).green()
    );

    Ok(())
}
