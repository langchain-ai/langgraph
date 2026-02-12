mod analytics;
mod commands;
mod config;
mod constants;
mod docker;
mod exec;
mod progress;
mod templates;
mod util;

use clap::{Parser, Subcommand};

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser)]
#[command(name = "langgraph", version = VERSION, about = "LangGraph CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Launch LangGraph API server with Docker
    Up {
        /// Path to configuration file declaring dependencies, graphs and environment variables
        #[arg(short, long, default_value = "langgraph.json")]
        config: String,

        /// Port to expose
        #[arg(short, long, default_value_t = 8123)]
        port: u16,

        /// Path to docker-compose.yml file with additional services
        #[arg(short, long)]
        docker_compose: Option<String>,

        /// Show detailed output
        #[arg(short, long)]
        verbose: bool,

        /// Restart on file changes using docker compose watch
        #[arg(short, long)]
        watch: bool,

        /// Recreate containers even if configuration hasn't changed
        #[arg(long)]
        recreate: bool,

        /// Pull latest images before running
        #[arg(long, default_value_t = true)]
        pull: bool,

        /// Wait for services to be healthy before returning
        #[arg(long)]
        wait: bool,

        /// Port to expose the debugger on
        #[arg(long)]
        debugger_port: Option<u16>,

        /// Base URL for the debugger
        #[arg(long)]
        debugger_base_url: Option<String>,

        /// Postgres connection URI
        #[arg(long)]
        postgres_uri: Option<String>,

        /// API version of the LangGraph server
        #[arg(long)]
        api_version: Option<String>,

        /// Pre-built image to use instead of building
        #[arg(long)]
        image: Option<String>,

        /// Base image for the LangGraph API server
        #[arg(long)]
        base_image: Option<String>,
    },

    /// Build LangGraph API server Docker image
    Build {
        /// Path to configuration file
        #[arg(short, long, default_value = "langgraph.json")]
        config: String,

        /// Tag for the docker image
        #[arg(short, long)]
        tag: String,

        /// Pull latest images before building
        #[arg(long, default_value_t = true)]
        pull: bool,

        /// Base image for the LangGraph API server
        #[arg(long)]
        base_image: Option<String>,

        /// API version of the LangGraph server
        #[arg(long)]
        api_version: Option<String>,

        /// Custom install command
        #[arg(long)]
        install_command: Option<String>,

        /// Custom build command
        #[arg(long)]
        build_command: Option<String>,

        /// Additional arguments to pass to docker build
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        docker_build_args: Vec<String>,
    },

    /// Generate a Dockerfile for the LangGraph API server
    Dockerfile {
        /// Path to save the generated Dockerfile
        save_path: String,

        /// Path to configuration file
        #[arg(short, long, default_value = "langgraph.json")]
        config: String,

        /// Add docker-compose.yml, .env, and .dockerignore files
        #[arg(long)]
        add_docker_compose: bool,

        /// Base image for the LangGraph API server
        #[arg(long)]
        base_image: Option<String>,

        /// API version of the LangGraph server
        #[arg(long)]
        api_version: Option<String>,
    },

    /// Run LangGraph API server in development mode
    Dev {
        /// Network interface to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port number
        #[arg(long, default_value_t = 2024)]
        port: u16,

        /// Disable automatic reloading
        #[arg(long)]
        no_reload: bool,

        /// Path to configuration file
        #[arg(long, default_value = "langgraph.json")]
        config: String,

        /// Max concurrent jobs per worker
        #[arg(long)]
        n_jobs_per_worker: Option<u32>,

        /// Skip opening browser
        #[arg(long)]
        no_browser: bool,

        /// Enable remote debugging on specified port
        #[arg(long)]
        debug_port: Option<u16>,

        /// Wait for debugger client to connect
        #[arg(long)]
        wait_for_client: bool,

        /// URL of LangGraph Studio
        #[arg(long)]
        studio_url: Option<String>,

        /// Allow synchronous I/O blocking operations
        #[arg(long)]
        allow_blocking: bool,

        /// Expose via public tunnel
        #[arg(long)]
        tunnel: bool,

        /// Log level for the API server
        #[arg(long, default_value = "WARNING")]
        server_log_level: String,
    },

    /// Create a new LangGraph project from a template
    New {
        /// Path to create the project
        path: Option<String>,

        /// Template to use
        #[arg(long)]
        template: Option<String>,
    },
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Up {
            config,
            port,
            docker_compose,
            verbose,
            watch,
            recreate,
            pull,
            wait,
            debugger_port,
            debugger_base_url,
            postgres_uri,
            api_version,
            image,
            base_image,
        } => commands::up::run(
            &config,
            port,
            docker_compose.as_deref(),
            verbose,
            watch,
            recreate,
            pull,
            wait,
            debugger_port,
            debugger_base_url.as_deref(),
            postgres_uri.as_deref(),
            api_version.as_deref(),
            image.as_deref(),
            base_image.as_deref(),
        ),
        Commands::Build {
            config,
            tag,
            pull,
            base_image,
            api_version,
            install_command,
            build_command,
            docker_build_args,
        } => commands::build_cmd::run(
            &config,
            &tag,
            pull,
            base_image.as_deref(),
            api_version.as_deref(),
            install_command.as_deref(),
            build_command.as_deref(),
            &docker_build_args,
        ),
        Commands::Dockerfile {
            save_path,
            config,
            add_docker_compose,
            base_image,
            api_version,
        } => commands::dockerfile::run(
            &save_path,
            &config,
            add_docker_compose,
            base_image.as_deref(),
            api_version.as_deref(),
        ),
        Commands::Dev {
            host,
            port,
            no_reload,
            config,
            n_jobs_per_worker,
            no_browser,
            debug_port,
            wait_for_client,
            studio_url,
            allow_blocking,
            tunnel,
            server_log_level,
        } => commands::dev::run(
            &host,
            port,
            no_reload,
            &config,
            n_jobs_per_worker,
            no_browser,
            debug_port,
            wait_for_client,
            studio_url.as_deref(),
            allow_blocking,
            tunnel,
            &server_log_level,
        ),
        Commands::New { path, template } => {
            commands::new::run(path.as_deref(), template.as_deref())
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
