use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::Path;

use console::style;
use dialoguer::{Input, Select};

/// A project template definition.
pub struct Template {
    pub name: &'static str,
    pub description: &'static str,
    pub python: &'static str,
    pub js: &'static str,
}

/// All available project templates.
pub const TEMPLATES: &[Template] = &[
    Template {
        name: "New LangGraph Project",
        description: "A simple, minimal chatbot with memory.",
        python: "https://github.com/langchain-ai/new-langgraph-project/archive/refs/heads/main.zip",
        js: "https://github.com/langchain-ai/new-langgraphjs-project/archive/refs/heads/main.zip",
    },
    Template {
        name: "ReAct Agent",
        description: "A simple agent that can be flexibly extended to many tools.",
        python: "https://github.com/langchain-ai/react-agent/archive/refs/heads/main.zip",
        js: "https://github.com/langchain-ai/react-agent-js/archive/refs/heads/main.zip",
    },
    Template {
        name: "Memory Agent",
        description: "A ReAct-style agent with an additional tool to store memories for use across conversational threads.",
        python: "https://github.com/langchain-ai/memory-agent/archive/refs/heads/main.zip",
        js: "https://github.com/langchain-ai/memory-agent-js/archive/refs/heads/main.zip",
    },
    Template {
        name: "Retrieval Agent",
        description: "An agent that includes a retrieval-based question-answering system.",
        python: "https://github.com/langchain-ai/retrieval-agent-template/archive/refs/heads/main.zip",
        js: "https://github.com/langchain-ai/retrieval-agent-template-js/archive/refs/heads/main.zip",
    },
    Template {
        name: "Data-enrichment Agent",
        description: "An agent that performs web searches and organizes its findings into a structured format.",
        python: "https://github.com/langchain-ai/data-enrichment/archive/refs/heads/main.zip",
        js: "https://github.com/langchain-ai/data-enrichment-js/archive/refs/heads/main.zip",
    },
];

/// Mapping from template ID (e.g., "react-agent-python") to (template index, language, url).
pub fn build_template_id_map() -> HashMap<String, (usize, &'static str, &'static str)> {
    let mut map = HashMap::new();
    for (idx, tmpl) in TEMPLATES.iter().enumerate() {
        let base = tmpl.name.to_lowercase().replace(' ', "-");
        map.insert(
            format!("{base}-python"),
            (idx, "python", tmpl.python),
        );
        map.insert(format!("{base}-js"), (idx, "js", tmpl.js));
    }
    map
}

/// Get a sorted list of all valid template IDs.
pub fn template_ids() -> Vec<String> {
    let map = build_template_id_map();
    let mut ids: Vec<String> = map.keys().cloned().collect();
    ids.sort();
    ids
}

/// Interactively choose a template. Returns the download URL.
fn choose_template() -> Result<String, String> {
    eprintln!("{}", style("Please select a template:").bold().yellow());
    let items: Vec<String> = TEMPLATES
        .iter()
        .map(|t| format!("{} - {}", t.name, t.description))
        .collect();

    let selection = Select::new()
        .with_prompt("Select a template")
        .items(&items)
        .default(0)
        .interact()
        .map_err(|e| format!("Template selection failed: {e}"))?;

    let tmpl = &TEMPLATES[selection];
    eprintln!(
        "\n{}",
        style(format!("You selected: {} - {}", tmpl.name, tmpl.description)).green()
    );

    let lang_items = vec!["Python", "JS/TS"];
    let lang_choice = Select::new()
        .with_prompt("Choose language")
        .items(&lang_items)
        .default(0)
        .interact()
        .map_err(|e| format!("Language selection failed: {e}"))?;

    let url = if lang_choice == 0 {
        tmpl.python
    } else {
        tmpl.js
    };
    Ok(url.to_string())
}

/// Download a zip archive from `url` and extract to `path`.
fn download_repo(url: &str, path: &str) -> Result<(), String> {
    eprintln!(
        "{}",
        style("Downloading repository as a ZIP archive...").yellow()
    );
    eprintln!("{}", style(format!("URL: {url}")).yellow());

    // Use blocking reqwest to download
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {e}"))?;

    let response = client
        .get(url)
        .send()
        .map_err(|e| format!("Failed to download repository: {e}"))?;

    if !response.status().is_success() {
        return Err(format!(
            "Failed to download repository. HTTP status: {}",
            response.status()
        ));
    }

    let bytes = response
        .bytes()
        .map_err(|e| format!("Failed to read response body: {e}"))?;

    let target_path = Path::new(path);
    if !target_path.exists() {
        std::fs::create_dir_all(target_path)
            .map_err(|e| format!("Failed to create directory {path}: {e}"))?;
    }

    let cursor = Cursor::new(bytes);
    let mut archive =
        zip::ZipArchive::new(cursor).map_err(|e| format!("Failed to open ZIP archive: {e}"))?;

    // Extract all files
    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| format!("Failed to read ZIP entry: {e}"))?;
        let name = file.name().to_string();

        // Strip the top-level directory (e.g., "repo-main/")
        let stripped = match name.split_once('/') {
            Some((_, rest)) if !rest.is_empty() => rest.to_string(),
            _ => continue, // Skip the top-level directory entry itself
        };

        let out_path = target_path.join(&stripped);

        if file.is_dir() {
            std::fs::create_dir_all(&out_path)
                .map_err(|e| format!("Failed to create directory {}: {e}", out_path.display()))?;
        } else {
            if let Some(parent) = out_path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    format!("Failed to create directory {}: {e}", parent.display())
                })?;
            }
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)
                .map_err(|e| format!("Failed to read file from ZIP: {e}"))?;
            std::fs::write(&out_path, &buf)
                .map_err(|e| format!("Failed to write {}: {e}", out_path.display()))?;
        }
    }

    eprintln!(
        "{}",
        style(format!("Downloaded and extracted repository to {path}")).green()
    );

    Ok(())
}

/// Create a new LangGraph project at the specified path using the chosen template.
///
/// If `path` is None, the user is prompted interactively.
/// If `template` is None, the user picks from an interactive menu.
pub fn create_new(path: Option<&str>, template: Option<&str>) -> Result<(), String> {
    // Prompt for path if not provided
    let path = match path {
        Some(p) => p.to_string(),
        None => {
            let input: String = Input::new()
                .with_prompt("Please specify the path to create the application")
                .default(".".to_string())
                .interact_text()
                .map_err(|e| format!("Input failed: {e}"))?;
            input
        }
    };

    let abs_path = std::path::Path::new(&path)
        .canonicalize()
        .unwrap_or_else(|_| std::path::PathBuf::from(&path));
    let abs_path_str = abs_path.to_string_lossy().to_string();

    // If the path doesn't exist yet, that's fine. But if it exists and is non-empty, abort.
    if abs_path.exists() {
        let entries = std::fs::read_dir(&abs_path)
            .map_err(|e| format!("Could not read directory {abs_path_str}: {e}"))?;
        if entries.count() > 0 {
            return Err(
                "The specified directory already exists and is not empty. \
                 Aborting to prevent overwriting files."
                    .to_string(),
            );
        }
    }

    // Get template URL either from command-line argument or through interactive selection
    let template_url = if let Some(tmpl_id) = template {
        let id_map = build_template_id_map();
        if let Some((_idx, _lang, url)) = id_map.get(tmpl_id) {
            url.to_string()
        } else {
            let ids = template_ids();
            let mut options = String::new();
            for id in &ids {
                let (_idx, _lang, _url) = id_map.get(id.as_str()).unwrap();
                let tmpl = &TEMPLATES[*_idx];
                options.push_str(&format!("- {id}: {}\n", tmpl.description));
            }
            return Err(format!(
                "Template '{tmpl_id}' not found.\n\
                 Please select from the available options:\n{options}"
            ));
        }
    } else {
        choose_template()?
    };

    // Download and extract the template
    download_repo(&template_url, &abs_path_str)?;

    eprintln!(
        "{}",
        style(format!("New project created at {abs_path_str}"))
            .green()
            .bold()
    );
    Ok(())
}
