use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Top-level config for langgraph-cli deployment.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub python_version: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_version: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,

    #[serde(rename = "_INTERNAL_docker_tag", skip_serializing_if = "Option::is_none")]
    pub internal_docker_tag: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_image: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_distro: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub pip_config_file: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub pip_installer: Option<String>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dockerfile_lines: Vec<String>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dependencies: Vec<String>,

    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub graphs: IndexMap<String, GraphSpec>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<EnvConfig>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth: Option<AuthConfig>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub encryption: Option<EncryptionConfig>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub http: Option<HttpConfig>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub webhooks: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpointer: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub ui: Option<IndexMap<String, String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub ui_config: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_pkg_tools: Option<KeepPkgTools>,
}

/// Graph specification: either a string path or a dict with a "path" key.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GraphSpec {
    Path(String),
    Dict(IndexMap<String, Value>),
}

impl GraphSpec {
    pub fn get_path(&self) -> Option<&str> {
        match self {
            GraphSpec::Path(s) => Some(s),
            GraphSpec::Dict(m) => m.get("path").and_then(|v| v.as_str()),
        }
    }

    pub fn set_path(&mut self, new_path: String) {
        match self {
            GraphSpec::Path(s) => *s = new_path,
            GraphSpec::Dict(m) => {
                m.insert("path".to_string(), Value::String(new_path));
            }
        }
    }
}

/// Environment config: either a dict of key-value pairs or a path to an env file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EnvConfig {
    Dict(IndexMap<String, String>),
    File(String),
}

impl Default for EnvConfig {
    fn default() -> Self {
        EnvConfig::Dict(IndexMap::new())
    }
}

/// Auth configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_studio_auth: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub openapi: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache: Option<Value>,
}

/// Encryption configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

/// HTTP configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub app: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_assistants: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_threads: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_runs: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_store: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_mcp: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_a2a: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_meta: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_ui: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_webhooks: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub cors: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub configurable_headers: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub logging_headers: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub middleware_order: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_custom_route_auth: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub mount_prefix: Option<String>,
}

/// Keep package tools config: either a boolean or a list of tool names.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum KeepPkgTools {
    Bool(bool),
    List(Vec<String>),
}
