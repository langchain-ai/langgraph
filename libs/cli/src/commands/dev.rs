use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use console::style;

use crate::analytics;
use crate::config::validate_config_file;

/// Run the LangGraph API server in development mode (in-memory, via Python subprocess).
///
/// This spawns `python -c "from langgraph_api.cli import run_server; ..."` as a subprocess,
/// passing the parsed config as arguments. For JS graphs, an error is returned since the
/// in-memory server doesn't support them in this CLI.
#[allow(clippy::too_many_arguments)]
pub fn run(
    host: &str,
    port: u16,
    no_reload: bool,
    config: &str,
    n_jobs_per_worker: Option<u32>,
    no_browser: bool,
    debug_port: Option<u16>,
    wait_for_client: bool,
    studio_url: Option<&str>,
    allow_blocking: bool,
    tunnel: bool,
    server_log_level: &str,
) -> Result<(), String> {
    // Fire-and-forget analytics
    let mut params = HashMap::new();
    params.insert("no_reload".to_string(), no_reload.to_string());
    params.insert("no_browser".to_string(), no_browser.to_string());
    params.insert("allow_blocking".to_string(), allow_blocking.to_string());
    params.insert("tunnel".to_string(), tunnel.to_string());
    analytics::log_command("dev", &params);

    // Validate config
    let config_path = Path::new(config);
    let config_json = validate_config_file(config_path)?;

    // Check for node_version -- in-mem server doesn't support JS graphs
    if config_json.node_version.is_some() {
        return Err(
            "In-mem server for JS graphs is not supported in this version of the LangGraph CLI. \
             Please use `npx @langchain/langgraph-cli` instead."
                .to_string(),
        );
    }

    // Build the graphs JSON
    let graphs_json = serde_json::to_string(&config_json.graphs)
        .map_err(|e| format!("Failed to serialize graphs: {e}"))?;

    // Build env JSON (optional)
    let env_json = config_json
        .env
        .as_ref()
        .map(|e| serde_json::to_string(e).unwrap_or_else(|_| "null".to_string()))
        .unwrap_or_else(|| "None".to_string());

    // Build store JSON (optional)
    let store_json = config_json
        .store
        .as_ref()
        .map(|s| serde_json::to_string(s).unwrap_or_else(|_| "null".to_string()))
        .unwrap_or_else(|| "None".to_string());

    // Build auth JSON (optional)
    let auth_json = config_json
        .auth
        .as_ref()
        .map(|a| serde_json::to_string(a).unwrap_or_else(|_| "null".to_string()))
        .unwrap_or_else(|| "None".to_string());

    // Build http JSON (optional)
    let http_json = config_json
        .http
        .as_ref()
        .map(|h| serde_json::to_string(h).unwrap_or_else(|_| "null".to_string()))
        .unwrap_or_else(|| "None".to_string());

    // Build ui JSON (optional)
    let ui_json = config_json
        .ui
        .as_ref()
        .map(|u| serde_json::to_string(u).unwrap_or_else(|_| "null".to_string()))
        .unwrap_or_else(|| "None".to_string());

    // Build ui_config JSON (optional)
    let ui_config_json = config_json
        .ui_config
        .as_ref()
        .map(|u| serde_json::to_string(u).unwrap_or_else(|_| "null".to_string()))
        .unwrap_or_else(|| "None".to_string());

    // Build webhooks JSON (optional)
    let webhooks_json = config_json
        .webhooks
        .as_ref()
        .map(|w| serde_json::to_string(w).unwrap_or_else(|_| "null".to_string()))
        .unwrap_or_else(|| "None".to_string());

    // Build n_jobs_per_worker
    let n_jobs_str = n_jobs_per_worker
        .map(|n| n.to_string())
        .unwrap_or_else(|| "None".to_string());

    // Build debug_port
    let debug_port_str = debug_port
        .map(|p| p.to_string())
        .unwrap_or_else(|| "None".to_string());

    // Build studio_url
    let studio_url_str = studio_url
        .map(|s| format!("\"{}\"", s.replace('"', "\\\"")))
        .unwrap_or_else(|| "None".to_string());

    // Construct the Python code to execute
    let python_code = format!(
        r#"
import sys, os, json
cwd = os.getcwd()
sys.path.append(cwd)
deps = json.loads('{deps_json}')
for dep in deps:
    import pathlib
    dep_path = pathlib.Path(cwd) / dep
    if dep_path.is_dir() and dep_path.exists():
        sys.path.append(str(dep_path))
from langgraph_api.cli import run_server
graphs = json.loads('{graphs}')
run_server(
    "{host}",
    {port},
    {reload},
    graphs,
    n_jobs_per_worker={n_jobs},
    open_browser={open_browser},
    debug_port={debug_port},
    env={env},
    store={store},
    wait_for_client={wait_for_client},
    auth={auth},
    http={http},
    ui={ui},
    ui_config={ui_config},
    webhooks={webhooks},
    studio_url={studio_url},
    allow_blocking={allow_blocking},
    tunnel={tunnel},
    server_level="{server_log_level}",
)
"#,
        deps_json = serde_json::to_string(&config_json.dependencies)
            .unwrap_or_else(|_| "[]".to_string())
            .replace('\'', "\\'"),
        graphs = graphs_json.replace('\'', "\\'"),
        host = host,
        port = port,
        reload = if no_reload { "False" } else { "True" },
        n_jobs = n_jobs_str,
        open_browser = if no_browser { "False" } else { "True" },
        debug_port = debug_port_str,
        env = env_json,
        store = store_json,
        wait_for_client = if wait_for_client { "True" } else { "False" },
        auth = auth_json,
        http = http_json,
        ui = ui_json,
        ui_config = ui_config_json,
        webhooks = webhooks_json,
        studio_url = studio_url_str,
        allow_blocking = if allow_blocking { "True" } else { "False" },
        tunnel = if tunnel { "True" } else { "False" },
        server_log_level = server_log_level,
    );

    eprintln!(
        "{}",
        style("Starting LangGraph API server in development mode...").green()
    );

    // Spawn Python subprocess
    let status = Command::new("python")
        .arg("-c")
        .arg(&python_code)
        .current_dir(
            config_path
                .parent()
                .unwrap_or_else(|| Path::new(".")),
        )
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                "Python not found. The `langgraph dev` command requires Python >= 3.11 with \
                 langgraph-cli[inmem] installed.\n\
                 Install with: pip install -U \"langgraph-cli[inmem]\""
                    .to_string()
            } else {
                format!("Failed to start Python: {e}")
            }
        })?;

    if !status.success() {
        let code = status.code().unwrap_or(1);
        if code == 130 {
            // User interrupted with Ctrl-C
            return Ok(());
        }
        return Err(format!("Development server exited with code {code}"));
    }

    Ok(())
}
