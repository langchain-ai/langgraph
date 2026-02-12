use console::style;

/// Remove empty lines from a string.
pub fn clean_empty_lines(input: &str) -> String {
    input
        .lines()
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Show warning if image_distro is not set to 'wolfi'.
pub fn warn_non_wolfi_distro(config: &serde_json::Value) {
    let image_distro = config
        .get("image_distro")
        .and_then(|v| v.as_str())
        .unwrap_or("debian");

    if image_distro != "wolfi" {
        eprintln!(
            "{}",
            style(
                "Warning: Security Recommendation: Consider switching to Wolfi Linux for enhanced security."
            )
            .yellow()
            .bold()
        );
        eprintln!(
            "{}",
            style(
                "   Wolfi is a security-oriented, minimal Linux distribution designed for containers."
            )
            .yellow()
        );
        eprintln!(
            "{}",
            style(
                "   To switch, add '\"image_distro\": \"wolfi\"' to your langgraph.json config file."
            )
            .yellow()
        );
        eprintln!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_empty_lines() {
        assert_eq!(clean_empty_lines("line1\n\nline2\n\nline3"), "line1\nline2\nline3");
        assert_eq!(clean_empty_lines("line1\nline2\nline3"), "line1\nline2\nline3");
        assert_eq!(clean_empty_lines("\n\n\n"), "");
        assert_eq!(clean_empty_lines(""), "");
    }
}
