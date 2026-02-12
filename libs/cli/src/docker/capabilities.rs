use std::process::Command;

/// Semantic version tuple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl Version {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
}

/// Type of Docker Compose installation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComposeType {
    Plugin,
    Standalone,
}

/// Docker capabilities detected on the system.
#[derive(Debug, Clone)]
pub struct DockerCapabilities {
    pub version_docker: Version,
    pub version_compose: Version,
    pub healthcheck_start_interval: bool,
    pub compose_type: ComposeType,
}

/// Parse a version string like "1.2.3", "v1.2.3-alpha", etc.
pub fn parse_version(version: &str) -> Version {
    let cleaned = version.trim();
    let parts: Vec<&str> = cleaned.split('.').collect();

    let parse_part = |s: &str| -> u32 {
        let s = s.trim_start_matches('v');
        let s = s.split('-').next().unwrap_or(s);
        let s = s.split('+').next().unwrap_or(s);
        s.parse().unwrap_or(0)
    };

    match parts.len() {
        1 => Version::new(parse_part(parts[0]), 0, 0),
        2 => Version::new(parse_part(parts[0]), parse_part(parts[1]), 0),
        _ => Version::new(
            parse_part(parts[0]),
            parse_part(parts[1]),
            parse_part(parts[2]),
        ),
    }
}

/// Check Docker capabilities on the system.
pub fn check_capabilities() -> Result<DockerCapabilities, String> {
    // Check docker is available
    if which::which("docker").is_err() {
        return Err("Docker not installed".to_string());
    }

    // Get docker info
    let output = Command::new("docker")
        .args(["info", "-f", "{{json .}}"])
        .output()
        .map_err(|_| "Docker not installed or not running".to_string())?;

    if !output.status.success() {
        return Err("Docker not installed or not running".to_string());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let info: serde_json::Value =
        serde_json::from_str(&stdout).map_err(|_| "Docker not installed or not running".to_string())?;

    let server_version = info
        .get("ServerVersion")
        .and_then(|v| v.as_str())
        .ok_or("Docker not running")?;

    if server_version.is_empty() {
        return Err("Docker not running".to_string());
    }

    // Try to find compose as plugin
    let (compose_version_str, compose_type) = if let Some(plugins) = info
        .get("ClientInfo")
        .and_then(|ci| ci.get("Plugins"))
        .and_then(|p| p.as_array())
    {
        if let Some(compose) = plugins
            .iter()
            .find(|p| p.get("Name").and_then(|n| n.as_str()) == Some("compose"))
        {
            let version = compose
                .get("Version")
                .and_then(|v| v.as_str())
                .unwrap_or("0.0.0");
            (version.to_string(), ComposeType::Plugin)
        } else {
            get_standalone_compose_version()?
        }
    } else {
        get_standalone_compose_version()?
    };

    let docker_version = parse_version(server_version);
    let compose_version = parse_version(&compose_version_str);

    Ok(DockerCapabilities {
        version_docker: docker_version,
        version_compose: compose_version,
        healthcheck_start_interval: docker_version >= Version::new(25, 0, 0),
        compose_type,
    })
}

fn get_standalone_compose_version() -> Result<(String, ComposeType), String> {
    if which::which("docker-compose").is_err() {
        return Err("Docker Compose not installed".to_string());
    }

    let output = Command::new("docker-compose")
        .args(["--version", "--short"])
        .output()
        .map_err(|_| "Docker Compose not installed".to_string())?;

    let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok((version, ComposeType::Standalone))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_version_basic() {
        assert_eq!(parse_version("1.2.3"), Version::new(1, 2, 3));
    }

    #[test]
    fn test_parse_version_with_v() {
        assert_eq!(parse_version("v1.2.3"), Version::new(1, 2, 3));
    }

    #[test]
    fn test_parse_version_with_prerelease() {
        assert_eq!(parse_version("1.2.3-alpha"), Version::new(1, 2, 3));
    }

    #[test]
    fn test_parse_version_with_build() {
        assert_eq!(parse_version("1.2.3+1"), Version::new(1, 2, 3));
    }

    #[test]
    fn test_parse_version_two_parts() {
        assert_eq!(parse_version("1.2"), Version::new(1, 2, 0));
    }

    #[test]
    fn test_parse_version_one_part() {
        assert_eq!(parse_version("1"), Version::new(1, 0, 0));
    }

    #[test]
    fn test_parse_version_complex() {
        assert_eq!(parse_version("v28.1.1+1"), Version::new(28, 1, 1));
    }

    #[test]
    fn test_parse_version_beta() {
        assert_eq!(
            parse_version("2.0.0-beta.1+exp.sha.5114f85"),
            Version::new(2, 0, 0)
        );
    }
}
