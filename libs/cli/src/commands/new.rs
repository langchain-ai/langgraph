use std::collections::HashMap;

use crate::analytics;
use crate::templates;

/// Create a new LangGraph project from a template.
pub fn run(path: Option<&str>, template: Option<&str>) -> Result<(), String> {
    // Fire-and-forget analytics
    let mut params = HashMap::new();
    if let Some(t) = template {
        params.insert("template".to_string(), t.to_string());
    }
    analytics::log_command("new", &params);

    templates::create_new(path, template)
}
