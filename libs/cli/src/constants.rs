pub const DEFAULT_CONFIG: &str = "langgraph.json";
pub const DEFAULT_PORT: u16 = 8123;

pub const MIN_NODE_VERSION: u32 = 20;
pub const DEFAULT_NODE_VERSION: u32 = 20;

pub const MIN_PYTHON_VERSION: (u32, u32) = (3, 11);
pub const DEFAULT_PYTHON_VERSION: &str = "3.11";

pub const DEFAULT_IMAGE_DISTRO: &str = "debian";

pub const BUILD_TOOLS: &[&str] = &["pip", "setuptools", "wheel"];

// Analytics
pub const SUPABASE_PUBLIC_API_KEY: &str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt6cmxwcG9qaW5wY3l5YWlweG5iIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTkyNTc1NzksImV4cCI6MjAzNDgzMzU3OX0.kkVOlLz3BxemA5nP-vat3K4qRtrDuO4SwZSR_htcX9c";
pub const SUPABASE_URL: &str = "https://kzrlppojinpcyyaipxnb.supabase.co";

pub const DEFAULT_POSTGRES_URI: &str =
    "postgres://postgres:postgres@langgraph-postgres:5432/postgres?sslmode=disable";

pub const VALID_DISTROS: &[&str] = &["debian", "wolfi", "bookworm"];
pub const VALID_PIP_INSTALLERS: &[&str] = &["auto", "pip", "uv"];

pub const RESERVED_PACKAGE_NAMES: &[&str] = &[
    "src",
    "langgraph-api",
    "langgraph_api",
    "langgraph",
    "langchain-core",
    "langchain_core",
    "pydantic",
    "orjson",
    "fastapi",
    "uvicorn",
    "psycopg",
    "httpx",
    "langsmith",
];
