use std::collections::HashMap;
use std::path::Path;

use console::style;

use crate::analytics;
use crate::config::validate_config_file;
use crate::docker::capabilities::check_capabilities;
use crate::docker::compose::compose_as_dict;
use crate::docker::compose::dict_to_yaml;
use crate::docker::dockerfile::config_to_docker;
use crate::util::warn_non_wolfi_distro;

/// Docker ignore file content.
fn get_docker_ignore_content() -> &'static str {
    "\
# Ignore node_modules and other dependency directories
node_modules
bower_components
vendor

# Ignore logs and temporary files
*.log
*.tmp
*.swp

# Ignore .env files and other environment files
.env
.env.*
*.local

# Ignore git-related files
.git
.gitignore

# Ignore Docker-related files and configs
.dockerignore
docker-compose.yml

# Ignore build and cache directories
dist
build
.cache
__pycache__

# Ignore IDE and editor configurations
.vscode
.idea
*.sublime-project
*.sublime-workspace
.DS_Store  # macOS-specific

# Ignore test and coverage files
coverage
*.coverage
*.test.js
*.spec.js
tests
"
}

/// Generate a Dockerfile for the LangGraph API server.
pub fn run(
    save_path: &str,
    config: &str,
    add_docker_compose: bool,
    base_image: Option<&str>,
    api_version: Option<&str>,
) -> Result<(), String> {
    // Fire-and-forget analytics
    let mut params = HashMap::new();
    params.insert(
        "add_docker_compose".to_string(),
        add_docker_compose.to_string(),
    );
    analytics::log_command("dockerfile", &params);

    let save_path = Path::new(save_path);
    let abs_save_path = if save_path.is_absolute() {
        save_path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_default()
            .join(save_path)
    };

    eprintln!(
        "{}",
        style(format!("Validating configuration at path: {config}")).yellow()
    );
    let config_path = Path::new(config);
    let mut config_json = validate_config_file(config_path)?;
    let config_value = serde_json::to_value(&config_json).unwrap();
    warn_non_wolfi_distro(&config_value);
    eprintln!("{}", style("Configuration validated!").green());

    eprintln!(
        "{}",
        style(format!(
            "Generating Dockerfile at {}",
            abs_save_path.display()
        ))
        .yellow()
    );

    let (dockerfile_content, additional_contexts) = config_to_docker(
        config_path,
        &mut config_json,
        base_image,
        api_version,
        None,
        None,
        None,
        false,
    )?;

    std::fs::write(&abs_save_path, &dockerfile_content)
        .map_err(|e| format!("Failed to write Dockerfile: {e}"))?;
    eprintln!("{}", style("Created: Dockerfile").green());

    if !additional_contexts.is_empty() {
        let ctx_str: Vec<String> = additional_contexts
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        eprintln!(
            "{}",
            style(format!(
                "Run docker build with these additional build contexts `--build-context {}`",
                ctx_str.join(",")
            ))
            .yellow()
        );
    }

    if add_docker_compose {
        let parent = abs_save_path.parent().unwrap_or(Path::new("."));

        // Write .dockerignore
        let dockerignore_path = parent.join(".dockerignore");
        std::fs::write(&dockerignore_path, get_docker_ignore_content())
            .map_err(|e| format!("Failed to write .dockerignore: {e}"))?;
        eprintln!("{}", style("Created: .dockerignore").green());

        // Generate docker-compose.yml
        let capabilities = check_capabilities()?;
        let mut compose_dict = compose_as_dict(
            &capabilities,
            8123,
            None,
            None,
            None,
            None,
            base_image,
            api_version,
        );

        // Add env_file and build context to langgraph-api service
        if let Some(crate::docker::compose::YamlValue::Dict(ref mut services)) =
            compose_dict.get_mut("services")
        {
            if let Some(crate::docker::compose::YamlValue::Dict(ref mut api_service)) =
                services.get_mut("langgraph-api")
            {
                api_service.insert(
                    "env_file".to_string(),
                    crate::docker::compose::YamlValue::List(vec![".env".to_string()]),
                );

                let dockerfile_name = abs_save_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                let mut build_config = indexmap::IndexMap::new();
                build_config.insert(
                    "context".to_string(),
                    crate::docker::compose::YamlValue::String(".".to_string()),
                );
                build_config.insert(
                    "dockerfile".to_string(),
                    crate::docker::compose::YamlValue::String(dockerfile_name),
                );
                api_service.insert(
                    "build".to_string(),
                    crate::docker::compose::YamlValue::Dict(build_config),
                );
            }
        }

        let compose_yaml = dict_to_yaml(&compose_dict, 0);
        let compose_path = parent.join("docker-compose.yml");
        std::fs::write(&compose_path, &compose_yaml)
            .map_err(|e| format!("Failed to write docker-compose.yml: {e}"))?;
        eprintln!("{}", style("Created: docker-compose.yml").green());

        // Create .env file if it doesn't exist
        let env_path = parent.join(".env");
        if !env_path.exists() {
            let env_content = "\
# Uncomment the following line to add your LangSmith API key
# LANGSMITH_API_KEY=your-api-key
# Or if you have a LangSmith Deployment license key, then uncomment the following line:
# LANGGRAPH_CLOUD_LICENSE_KEY=your-license-key
# Add any other environment variables go below...
";
            std::fs::write(&env_path, env_content)
                .map_err(|e| format!("Failed to write .env: {e}"))?;
            eprintln!("{}", style("Created: .env").green());
        } else {
            eprintln!(
                "{}",
                style("Skipped: .env. It already exists!").yellow()
            );
        }
    }

    eprintln!(
        "{}",
        style(format!(
            "Files generated successfully at path {}!",
            abs_save_path
                .parent()
                .unwrap_or(Path::new("."))
                .display()
        ))
        .cyan()
        .bold()
    );

    Ok(())
}
