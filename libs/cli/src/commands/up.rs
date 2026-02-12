use std::collections::HashMap;
use std::path::Path;

use console::style;

use crate::analytics;
use crate::config::docker_tag::{default_base_image, docker_tag};
use crate::config::validate_config_file;
use crate::docker::capabilities::{check_capabilities, ComposeType};
use crate::docker::compose::compose;
use crate::docker::dockerfile::config_to_compose;
use crate::exec::{run_command, run_command_streaming};
use crate::progress::Progress;
use crate::util::warn_non_wolfi_distro;

/// Launch LangGraph API server with Docker Compose.
#[allow(clippy::too_many_arguments)]
pub fn run(
    config: &str,
    port: u16,
    docker_compose: Option<&str>,
    verbose: bool,
    watch: bool,
    recreate: bool,
    pull: bool,
    wait: bool,
    debugger_port: Option<u16>,
    debugger_base_url: Option<&str>,
    postgres_uri: Option<&str>,
    api_version: Option<&str>,
    image: Option<&str>,
    base_image: Option<&str>,
) -> Result<(), String> {
    // Fire-and-forget analytics
    let mut params = HashMap::new();
    params.insert("verbose".to_string(), verbose.to_string());
    params.insert("watch".to_string(), watch.to_string());
    params.insert("recreate".to_string(), recreate.to_string());
    params.insert("pull".to_string(), pull.to_string());
    params.insert("wait".to_string(), wait.to_string());
    analytics::log_command("up", &params);

    eprintln!("{}", style("Starting LangGraph API server...").green());
    eprintln!(
        "For local dev, requires env var LANGSMITH_API_KEY with access to LangSmith Deployment.\n\
         For production use, requires a license key in env var LANGGRAPH_CLOUD_LICENSE_KEY."
    );

    let progress = Progress::new("Pulling...");

    // Validate config
    let config_path = Path::new(config);
    let mut config_json = validate_config_file(config_path)?;
    let config_value = serde_json::to_value(&config_json).unwrap();
    warn_non_wolfi_distro(&config_value);

    // Check docker capabilities
    let capabilities = check_capabilities()?;

    // Pull latest images if requested
    if pull {
        let tag = docker_tag(&config_json, base_image, api_version);
        progress.set_message("Pulling...");
        run_command("docker", &["pull", &tag], None, verbose)?;
    }

    // Generate compose YAML
    let debugger_base_url_resolved = debugger_base_url
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("http://127.0.0.1:{port}"));

    let mut compose_stdin = compose(
        &capabilities,
        port,
        debugger_port,
        Some(&debugger_base_url_resolved),
        postgres_uri,
        image,
        base_image,
        api_version,
    );

    // Append config-to-compose output (build instructions, env, watch sections)
    let base_img = base_image
        .map(|s| s.to_string())
        .unwrap_or_else(|| default_base_image(&config_json));
    let compose_config = config_to_compose(
        config_path,
        &mut config_json,
        Some(&base_img),
        api_version,
        image,
        watch,
    )?;
    compose_stdin.push_str(&compose_config);

    // Build docker compose args
    let mut args: Vec<String> = Vec::new();
    args.push("--project-directory".to_string());
    args.push(
        config_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_string_lossy()
            .to_string(),
    );

    if let Some(dc) = docker_compose {
        args.push("-f".to_string());
        args.push(dc.to_string());
    }

    // Read compose from stdin
    args.push("-f".to_string());
    args.push("-".to_string());

    // Add up + options
    args.push("up".to_string());
    args.push("--remove-orphans".to_string());

    if recreate {
        args.push("--force-recreate".to_string());
        args.push("--renew-anon-volumes".to_string());
        // Try to remove the volume, ignore errors
        let _ = run_command("docker", &["volume", "rm", "langgraph-data"], None, false);
    }

    if watch {
        args.push("--watch".to_string());
    }

    if wait {
        args.push("--wait".to_string());
    } else {
        args.push("--abort-on-container-exit".to_string());
    }

    progress.set_message("Building...");

    // Determine compose command
    let compose_cmd = match capabilities.compose_type {
        ComposeType::Plugin => vec!["docker", "compose"],
        ComposeType::Standalone => vec!["docker-compose"],
    };

    // Build final command args
    let mut cmd_args: Vec<&str> = Vec::new();
    if compose_cmd.len() > 1 {
        // "docker compose ..."
        cmd_args.extend_from_slice(&compose_cmd[1..]);
    }
    for a in &args {
        cmd_args.push(a.as_str());
    }

    progress.finish();

    // Run docker compose with streaming output
    run_command_streaming(compose_cmd[0], &cmd_args, Some(&compose_stdin), verbose)?;

    Ok(())
}
