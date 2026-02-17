use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use console::style;

use crate::analytics;
use crate::config::validate_config_file;

/// Find the Python interpreter, preferring python3 over python.
fn find_python() -> Result<String, String> {
    for candidate in &["python3", "python"] {
        if which::which(candidate).is_ok() {
            return Ok(candidate.to_string());
        }
    }
    Err(
        "Python not found. The `langgraph dev` command requires Python >= 3.11 with \
         langgraph-cli[inmem] installed.\n\
         Install with: pip install -U \"langgraph-cli[inmem]\""
            .to_string(),
    )
}

/// Run the LangGraph API server in development mode (in-memory, via Python subprocess).
///
/// This passes the config as a JSON object via stdin to a Python bootstrap script,
/// avoiding any string interpolation into Python source code.
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

    let python = find_python()?;

    // Pre-check that langgraph_api is importable
    let check = Command::new(&python)
        .args(["-c", "from langgraph_api.cli import run_server"])
        .output();

    match check {
        Ok(output) if !output.status.success() => {
            return Err(
                "Required package 'langgraph-api' is not installed.\n\
                 Please install it with:\n\n\
                     pip install -U \"langgraph-cli[inmem]\"\n\n\
                 Note: The in-mem server requires Python 3.11 or higher."
                    .to_string(),
            );
        }
        Err(_) => {
            return Err(format!(
                "Failed to run {python}. The `langgraph dev` command requires Python >= 3.11 with \
                 langgraph-cli[inmem] installed.\n\
                 Install with: pip install -U \"langgraph-cli[inmem]\""
            ));
        }
        _ => {}
    }

    // Build a JSON config object to pass via stdin.
    // This avoids interpolating user data into Python source code.
    let dev_config = serde_json::json!({
        "host": host,
        "port": port,
        "reload": !no_reload,
        "open_browser": !no_browser,
        "wait_for_client": wait_for_client,
        "allow_blocking": allow_blocking,
        "tunnel": tunnel,
        "server_log_level": server_log_level,
        "dependencies": config_json.dependencies,
        "graphs": config_json.graphs,
        "n_jobs_per_worker": n_jobs_per_worker,
        "debug_port": debug_port,
        "studio_url": studio_url,
        "env": config_json.env,
        "store": config_json.store,
        "auth": config_json.auth,
        "http": config_json.http,
        "ui": config_json.ui,
        "ui_config": config_json.ui_config,
        "webhooks": config_json.webhooks,
    });

    // Python bootstrap: reads JSON from stdin, calls run_server
    let python_code = r#"
import sys, os, json, pathlib
config = json.loads(sys.stdin.read())
cwd = os.getcwd()
sys.path.append(cwd)
for dep in config.get('dependencies', []):
    dep_path = pathlib.Path(cwd) / dep
    if dep_path.is_dir() and dep_path.exists():
        sys.path.append(str(dep_path))
from langgraph_api.cli import run_server
run_server(
    config['host'],
    config['port'],
    config['reload'],
    config['graphs'],
    n_jobs_per_worker=config.get('n_jobs_per_worker'),
    open_browser=config['open_browser'],
    debug_port=config.get('debug_port'),
    env=config.get('env'),
    store=config.get('store'),
    wait_for_client=config['wait_for_client'],
    auth=config.get('auth'),
    http=config.get('http'),
    ui=config.get('ui'),
    ui_config=config.get('ui_config'),
    webhooks=config.get('webhooks'),
    studio_url=config.get('studio_url'),
    allow_blocking=config['allow_blocking'],
    tunnel=config['tunnel'],
    server_level=config['server_log_level'],
)
"#;

    eprintln!(
        "{}",
        style("Starting LangGraph API server in development mode...").green()
    );

    // Spawn Python subprocess with config on stdin
    let mut child = Command::new(&python)
        .arg("-c")
        .arg(python_code)
        .current_dir(
            config_path
                .parent()
                .unwrap_or_else(|| Path::new(".")),
        )
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .map_err(|e| format!("Failed to start Python: {e}"))?;

    // Write JSON config to stdin
    if let Some(ref mut stdin) = child.stdin {
        use std::io::Write;
        let json_bytes = dev_config.to_string();
        stdin
            .write_all(json_bytes.as_bytes())
            .map_err(|e| format!("Failed to write config to Python stdin: {e}"))?;
    }
    // Drop stdin to signal EOF
    drop(child.stdin.take());

    let status = child
        .wait()
        .map_err(|e| format!("Failed to wait for Python: {e}"))?;

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
