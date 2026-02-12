use std::collections::HashMap;

use crate::constants::{SUPABASE_PUBLIC_API_KEY, SUPABASE_URL};

/// Fire-and-forget telemetry: log a CLI command invocation.
///
/// Spawns a background thread that POSTs anonymized usage data to Supabase.
/// Respects the `LANGGRAPH_CLI_NO_ANALYTICS` environment variable -- if set to "1",
/// no data is sent.
pub fn log_command(command: &str, params: &HashMap<String, String>) {
    if std::env::var("LANGGRAPH_CLI_NO_ANALYTICS").as_deref() == Ok("1") {
        return;
    }

    let os_name = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();
    let cli_version = env!("CARGO_PKG_VERSION").to_string();
    let command = command.to_string();
    let params = params.clone();

    std::thread::spawn(move || {
        let data = serde_json::json!({
            "os": os_name,
            "os_version": arch,
            "python_version": "rust",
            "cli_version": cli_version,
            "cli_command": command,
            "params": params,
        });

        let url = format!("{SUPABASE_URL}/rest/v1/logs");

        // Use a blocking reqwest client in this thread
        let client = match reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
        {
            Ok(c) => c,
            Err(_) => return,
        };

        let _ = client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("apikey", SUPABASE_PUBLIC_API_KEY)
            .header("User-Agent", "Mozilla/5.0")
            .body(data.to_string())
            .send();
    });
}
