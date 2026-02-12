use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use indexmap::IndexMap;

use super::schema::Config;
use crate::constants::RESERVED_PACKAGE_NAMES;

/// Container for referencing and managing local Python dependencies.
#[derive(Debug, Clone)]
pub struct LocalDeps {
    /// (host_requirements_path, container_requirements_path)
    pub pip_reqs: Vec<(PathBuf, String)>,
    /// host_path -> (dependency_string, container_package_name)
    pub real_pkgs: IndexMap<PathBuf, (String, String)>,
    /// host_path -> (dependency_string, container_path)
    pub faux_pkgs: IndexMap<PathBuf, (String, String)>,
    /// If "." is in dependencies, use it as working_dir
    pub working_dir: Option<String>,
    /// Directories outside the config parent that need additional Docker build contexts
    pub additional_contexts: Vec<PathBuf>,
}

impl Default for LocalDeps {
    fn default() -> Self {
        Self {
            pip_reqs: Vec::new(),
            real_pkgs: IndexMap::new(),
            faux_pkgs: IndexMap::new(),
            working_dir: None,
            additional_contexts: Vec::new(),
        }
    }
}

/// Assemble local dependencies from config.
pub fn assemble_local_deps(
    config_path: &Path,
    config: &Config,
) -> Result<LocalDeps, String> {
    let config_path = config_path
        .canonicalize()
        .map_err(|e| format!("Could not resolve config path: {e}"))?;
    let config_parent = config_path.parent().unwrap();

    let mut reserved: HashSet<String> = RESERVED_PACKAGE_NAMES
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut counter: HashMap<String, usize> = HashMap::new();

    let check_reserved = |name: &str, ref_str: &str, reserved: &mut HashSet<String>| -> Result<(), String> {
        if reserved.contains(name) {
            return Err(format!(
                "Package name '{name}' used in local dep '{ref_str}' is reserved. Rename the directory."
            ));
        }
        reserved.insert(name.to_string());
        Ok(())
    };

    let mut pip_reqs = Vec::new();
    let mut real_pkgs = IndexMap::new();
    let mut faux_pkgs = IndexMap::new();
    let mut working_dir: Option<String> = None;
    let mut additional_contexts = Vec::new();

    for local_dep in &config.dependencies {
        if !local_dep.starts_with('.') {
            continue;
        }

        let resolved = (config_parent.join(local_dep))
            .canonicalize()
            .map_err(|_| format!("Could not find local dependency: {}", config_parent.join(local_dep).display()))?;

        if !resolved.exists() {
            return Err(format!("Could not find local dependency: {}", resolved.display()));
        }
        if !resolved.is_dir() {
            return Err(format!(
                "Local dependency must be a directory: {}",
                resolved.display()
            ));
        }

        if resolved != config_parent && !resolved.starts_with(config_parent) {
            additional_contexts.push(resolved.clone());
        }

        let files: Vec<String> = std::fs::read_dir(&resolved)
            .map_err(|e| format!("Could not read directory {}: {e}", resolved.display()))?
            .filter_map(|entry| entry.ok().map(|e| e.file_name().to_string_lossy().to_string()))
            .collect();

        if files.contains(&"pyproject.toml".to_string())
            || files.contains(&"setup.py".to_string())
        {
            // Real package
            let dir_name = resolved
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string();
            let count = counter.entry(dir_name.clone()).or_insert(0);
            let container_name = if *count > 0 {
                format!("{}_{}", dir_name, count)
            } else {
                dir_name.clone()
            };
            *count += 1;

            real_pkgs.insert(resolved.clone(), (local_dep.clone(), container_name.clone()));

            if local_dep == "." {
                working_dir = Some(format!("/deps/{container_name}"));
            }
        } else {
            // Faux package
            let dir_name = resolved
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string();

            if files.contains(&"__init__.py".to_string()) {
                // Flat layout
                if dir_name.contains('-') {
                    return Err(format!(
                        "Package name '{dir_name}' contains a hyphen. \
                         Rename the directory to use it as flat-layout package."
                    ));
                }
                check_reserved(&dir_name, local_dep, &mut reserved)?;
                let container_path = format!("/deps/outer-{dir_name}/{dir_name}");
                faux_pkgs.insert(resolved.clone(), (local_dep.clone(), container_path.clone()));
                if local_dep == "." {
                    working_dir = Some(container_path);
                }
            } else {
                // Src layout
                let container_path = format!("/deps/outer-{dir_name}/src");

                for file in &files {
                    let rfile = resolved.join(file);
                    if rfile.is_dir() && file != "__pycache__" && !file.starts_with('.') {
                        if let Ok(entries) = std::fs::read_dir(&rfile) {
                            for subentry in entries.flatten() {
                                let subname = subentry.file_name().to_string_lossy().to_string();
                                if subname.ends_with(".py") {
                                    check_reserved(file, local_dep, &mut reserved)?;
                                    break;
                                }
                            }
                        }
                    }
                }

                faux_pkgs.insert(resolved.clone(), (local_dep.clone(), container_path.clone()));
                if local_dep == "." {
                    working_dir = Some(container_path);
                }
            }

            // Check for requirements.txt
            if files.contains(&"requirements.txt".to_string()) {
                let rfile = resolved.join("requirements.txt");
                let container_req_path = if let Some((_, ref cp)) = faux_pkgs.get(&resolved) {
                    format!("{cp}/requirements.txt")
                } else {
                    format!("/deps/outer-{dir_name}/requirements.txt")
                };
                pip_reqs.push((rfile, container_req_path));
            }
        }
    }

    Ok(LocalDeps {
        pip_reqs,
        real_pkgs,
        faux_pkgs,
        working_dir,
        additional_contexts,
    })
}
