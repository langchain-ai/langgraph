use std::path::Path;

use super::local_deps::LocalDeps;
use super::schema::{Config, GraphSpec};

/// Remap each graph's import path to the correct in-container path.
pub fn update_graph_paths(
    config_path: &Path,
    config: &mut Config,
    local_deps: &LocalDeps,
) -> Result<(), String> {
    let config_parent = config_path.parent().unwrap();
    let graph_ids: Vec<String> = config.graphs.keys().cloned().collect();

    for graph_id in graph_ids {
        let import_str = {
            let spec = config.graphs.get(&graph_id).unwrap();
            match spec {
                GraphSpec::Path(s) => s.clone(),
                GraphSpec::Dict(m) => {
                    if let Some(path_val) = m.get("path") {
                        path_val
                            .as_str()
                            .ok_or_else(|| {
                                format!("Graph '{graph_id}' path must be a string")
                            })?
                            .to_string()
                    } else {
                        return Err(format!(
                            "Graph '{graph_id}' must contain a 'path' key if it is a dictionary."
                        ));
                    }
                }
            }
        };

        let (module_str, attr_str) = match import_str.split_once(':') {
            Some((m, a)) if !m.is_empty() && !a.is_empty() => (m, a),
            _ => {
                return Err(format!(
                    "Import string \"{import_str}\" must be in format \"<module>:<attribute>\"."
                ));
            }
        };

        // Check for file path (contains / or \)
        if module_str.contains('/') || module_str.contains('\\') {
            let resolved = config_parent
                .join(module_str)
                .canonicalize()
                .map_err(|_| format!("Could not find local module: {}", config_parent.join(module_str).display()))?;

            if !resolved.exists() {
                return Err(format!("Could not find local module: {}", resolved.display()));
            }
            if !resolved.is_file() {
                return Err(format!("Local module must be a file: {}", resolved.display()));
            }

            let mut new_module = None;

            // Check real packages
            for (path, (_, container_name)) in &local_deps.real_pkgs {
                if resolved.starts_with(path) {
                    if let Ok(relative) = resolved.strip_prefix(path) {
                        let container_path =
                            format!("/deps/{}/{}", container_name, relative.to_string_lossy().replace('\\', "/"));
                        new_module = Some(container_path);
                        break;
                    }
                }
            }

            // Check faux packages
            if new_module.is_none() {
                for (faux_path, (_, destpath)) in &local_deps.faux_pkgs {
                    if resolved.starts_with(faux_path) {
                        if let Ok(relative) = resolved.strip_prefix(faux_path) {
                            new_module = Some(format!(
                                "{}/{}",
                                destpath,
                                relative.to_string_lossy().replace('\\', "/")
                            ));
                            break;
                        }
                    }
                }
            }

            if let Some(new_mod) = new_module {
                let new_path = format!("{new_mod}:{attr_str}");
                config.graphs.get_mut(&graph_id).unwrap().set_path(new_path);
            } else {
                return Err(format!(
                    "Module '{import_str}' not found in 'dependencies' list. \
                     Add its containing package to 'dependencies' list."
                ));
            }
        }
    }
    Ok(())
}

/// Update auth.path to use Docker container paths.
pub fn update_auth_path(
    config_path: &Path,
    config: &mut Config,
    local_deps: &LocalDeps,
) -> Result<(), String> {
    let auth_conf = match config.auth.as_mut() {
        Some(a) => a,
        None => return Ok(()),
    };
    let path_str = match auth_conf.path.as_ref() {
        Some(p) => p.clone(),
        None => return Ok(()),
    };

    let (module_str, attr_str) = match path_str.split_once(':') {
        Some((m, a)) => (m, a),
        None => return Ok(()),
    };

    if !module_str.starts_with('.') {
        return Ok(());
    }

    let config_parent = config_path.parent().unwrap();
    let resolved = config_parent
        .join(module_str)
        .canonicalize()
        .map_err(|_| format!("Auth file not found: {} (from {path_str})", config_parent.join(module_str).display()))?;

    if !resolved.is_file() {
        return Err(format!("Auth path must be a file: {}", resolved.display()));
    }

    // Check faux packages first
    for (faux_path, (_, destpath)) in &local_deps.faux_pkgs {
        if resolved.starts_with(faux_path) {
            if let Ok(relative) = resolved.strip_prefix(faux_path) {
                auth_conf.path = Some(format!(
                    "{}/{}:{attr_str}",
                    destpath,
                    relative.to_string_lossy()
                ));
                return Ok(());
            }
        }
    }

    // Check real packages
    for (real_path, _) in &local_deps.real_pkgs {
        if resolved.starts_with(real_path) {
            if let Ok(relative) = resolved.strip_prefix(real_path) {
                let dir_name = real_path.file_name().unwrap().to_string_lossy();
                auth_conf.path = Some(format!(
                    "/deps/{}/{}:{attr_str}",
                    dir_name,
                    relative.to_string_lossy()
                ));
                return Ok(());
            }
        }
    }

    Err(format!(
        "Auth file '{}' not covered by dependencies.\n\
         Add its parent directory to the 'dependencies' array in your config.",
        resolved.display()
    ))
}

/// Update encryption.path to use Docker container paths.
pub fn update_encryption_path(
    config_path: &Path,
    config: &mut Config,
    local_deps: &LocalDeps,
) -> Result<(), String> {
    let encryption_conf = match config.encryption.as_mut() {
        Some(e) => e,
        None => return Ok(()),
    };
    let path_str = match encryption_conf.path.as_ref() {
        Some(p) => p.clone(),
        None => return Ok(()),
    };

    let (module_str, attr_str) = match path_str.split_once(':') {
        Some((m, a)) => (m, a),
        None => return Ok(()),
    };

    if !module_str.starts_with('.') {
        return Ok(());
    }

    let config_parent = config_path.parent().unwrap();
    let resolved = config_parent
        .join(module_str)
        .canonicalize()
        .map_err(|_| {
            format!(
                "Encryption file not found: {} (from {path_str})",
                config_parent.join(module_str).display()
            )
        })?;

    if !resolved.is_file() {
        return Err(format!(
            "Encryption path must be a file: {}",
            resolved.display()
        ));
    }

    for (faux_path, (_, destpath)) in &local_deps.faux_pkgs {
        if resolved.starts_with(faux_path) {
            if let Ok(relative) = resolved.strip_prefix(faux_path) {
                encryption_conf.path = Some(format!(
                    "{}/{}:{attr_str}",
                    destpath,
                    relative.to_string_lossy()
                ));
                return Ok(());
            }
        }
    }

    for (real_path, _) in &local_deps.real_pkgs {
        if resolved.starts_with(real_path) {
            if let Ok(relative) = resolved.strip_prefix(real_path) {
                let dir_name = real_path.file_name().unwrap().to_string_lossy();
                encryption_conf.path = Some(format!(
                    "/deps/{}/{}:{attr_str}",
                    dir_name,
                    relative.to_string_lossy()
                ));
                return Ok(());
            }
        }
    }

    Err(format!(
        "Encryption file '{}' not covered by dependencies.\n\
         Add its parent directory to the 'dependencies' array in your config.",
        resolved.display()
    ))
}

/// Update the HTTP app path to point to the correct location in the Docker container.
pub fn update_http_app_path(
    config_path: &Path,
    config: &mut Config,
    local_deps: &LocalDeps,
) -> Result<(), String> {
    let http_config = match config.http.as_mut() {
        Some(h) => h,
        None => return Ok(()),
    };
    let app_str = match http_config.app.as_ref() {
        Some(a) => a.clone(),
        None => return Ok(()),
    };

    let (module_str, attr_str) = match app_str.split_once(':') {
        Some((m, a)) if !m.is_empty() && !a.is_empty() => (m, a),
        _ => {
            return Err(format!(
                "Import string \"{app_str}\" must be in format \"<module>:<attribute>\"."
            ));
        }
    };

    if !module_str.contains('/') && !module_str.contains('\\') {
        return Ok(());
    }

    let config_parent = config_path.parent().unwrap();
    let resolved = config_parent
        .join(module_str)
        .canonicalize()
        .map_err(|_| format!("Could not find HTTP app module: {}", config_parent.join(module_str).display()))?;

    if !resolved.is_file() {
        return Err(format!(
            "HTTP app module must be a file: {}",
            resolved.display()
        ));
    }

    // Check real packages
    for (path, (_, _name)) in &local_deps.real_pkgs {
        if resolved.starts_with(path) {
            if let Ok(relative) = resolved.strip_prefix(path) {
                let dir_name = path.file_name().unwrap().to_string_lossy();
                http_config.app = Some(format!(
                    "/deps/{}/{}:{attr_str}",
                    dir_name,
                    relative.to_string_lossy().replace('\\', "/")
                ));
                return Ok(());
            }
        }
    }

    // Check faux packages
    for (faux_path, (_, destpath)) in &local_deps.faux_pkgs {
        if resolved.starts_with(faux_path) {
            if let Ok(relative) = resolved.strip_prefix(faux_path) {
                http_config.app = Some(format!(
                    "{}/{}:{attr_str}",
                    destpath,
                    relative.to_string_lossy().replace('\\', "/")
                ));
                return Ok(());
            }
        }
    }

    Err(format!(
        "HTTP app module '{app_str}' not found in 'dependencies' list. \
         Add its containing package to 'dependencies' list."
    ))
}
