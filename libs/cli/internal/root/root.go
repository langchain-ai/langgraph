package root

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	osexec "os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/langchain-ai/langgraph/libs/cli/internal/config"
	"github.com/langchain-ai/langgraph/libs/cli/internal/deploy"
	"github.com/langchain-ai/langgraph/libs/cli/internal/docker"
	lgexec "github.com/langchain-ai/langgraph/libs/cli/internal/exec"
	"github.com/langchain-ai/langgraph/libs/cli/internal/templates"
	"github.com/langchain-ai/langgraph/libs/cli/internal/version"
)

var runPythonSubprocess = func(
	pythonExe string,
	args []string,
	stdout io.Writer,
	stderr io.Writer,
) error {
	cmd := osexec.Command(pythonExe, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	return cmd.Run()
}

const helpText = `Usage: langgraph [OPTIONS] COMMAND [ARGS]...

  LangGraph CLI

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  build       Build LangGraph API server Docker image.
  deploy      Build and deploy to LangSmith.
  dev         Run LangGraph API server in development mode.
  dockerfile  Generate a Dockerfile for the LangGraph API server.
  new         Create a new LangGraph project from a template.
  up          Launch LangGraph API server.
  validate    Validate the LangGraph configuration file.`

func Run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		_, _ = fmt.Fprintln(stdout, helpText)
		return 0
	}

	switch args[0] {
	case "help", "--help", "-h":
		_, _ = fmt.Fprintln(stdout, helpText)
		return 0
	case "version", "--version", "-V":
		_, _ = fmt.Fprintf(
			stdout,
			"langgraph %s (commit: %s, built: %s)\n",
			version.Version,
			version.Commit,
			version.Date,
		)
		return 0
	case "validate":
		return runValidate(args[1:], stdout, stderr)
	case "build":
		return runBuild(args[1:], stdout, stderr)
	case "dockerfile":
		return runDockerfile(args[1:], stdout, stderr)
	case "up":
		return runUp(args[1:], stdout, stderr)
	case "dev":
		return runDev(args[1:], stdout, stderr)
	case "new":
		return runNew(args[1:], stdout, stderr)
	case "deploy":
		return runDeploy(args[1:], stdout, stderr)
	default:
		_, _ = fmt.Fprintf(
			stderr,
			"langgraph: %q is not a langgraph command. See 'langgraph --help'.\n",
			args[0],
		)
		return 1
	}
}

// ---------------------------------------------------------------------------
// ANSI colors
// ---------------------------------------------------------------------------

const (
	colorReset  = "\033[0m"
	colorRed    = "\033[31m"
	colorGreen  = "\033[32m"
	colorYellow = "\033[33m"
	colorCyan   = "\033[36m"
)

func errPrint(w io.Writer, msg string) {
	_, _ = fmt.Fprintf(w, "%sError: %s%s\n", colorRed, msg, colorReset)
}

// ---------------------------------------------------------------------------
// Common flag parsing helpers
// ---------------------------------------------------------------------------

type commonFlags struct {
	configPath        string
	baseImage         string
	apiVersion        string
	tag               string
	port              int
	pull              bool
	verbose           bool
	watch             bool
	engineRuntimeMode string
	installCommand    string
	buildCommand      string
	dockerCompose     string
	postgresURI       string
	debuggerPort      int
	debuggerBaseURL   string
	image             string
	wait              bool
	passthrough       []string
}

func newCommonFlags() commonFlags {
	return commonFlags{
		configPath:        "langgraph.json",
		port:              8123,
		pull:              true,
		engineRuntimeMode: "combined_queue_worker",
	}
}

// parseFlags is a minimal flag parser. Unknown flags after "--" or positional
// args are collected into passthrough.
func parseFlags(args []string, flags *commonFlags) []string {
	var positional []string
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "-c", "--config":
			if i+1 < len(args) {
				flags.configPath = args[i+1]
				i++
			}
		case "-t", "--tag":
			if i+1 < len(args) {
				flags.tag = args[i+1]
				i++
			}
		case "-p", "--port":
			if i+1 < len(args) {
				if n, err := strconv.Atoi(args[i+1]); err == nil {
					flags.port = n
				}
				i++
			}
		case "--base-image":
			if i+1 < len(args) {
				flags.baseImage = args[i+1]
				i++
			}
		case "--api-version":
			if i+1 < len(args) {
				flags.apiVersion = args[i+1]
				i++
			}
		case "--engine-runtime-mode":
			if i+1 < len(args) {
				flags.engineRuntimeMode = args[i+1]
				i++
			}
		case "--install-command":
			if i+1 < len(args) {
				flags.installCommand = args[i+1]
				i++
			}
		case "--build-command":
			if i+1 < len(args) {
				flags.buildCommand = args[i+1]
				i++
			}
		case "--docker-compose", "-d":
			if i+1 < len(args) {
				flags.dockerCompose = args[i+1]
				i++
			}
		case "--postgres-uri":
			if i+1 < len(args) {
				flags.postgresURI = args[i+1]
				i++
			}
		case "--debugger-port":
			if i+1 < len(args) {
				if n, err := strconv.Atoi(args[i+1]); err == nil {
					flags.debuggerPort = n
				}
				i++
			}
		case "--debugger-base-url":
			if i+1 < len(args) {
				flags.debuggerBaseURL = args[i+1]
				i++
			}
		case "--image":
			if i+1 < len(args) {
				flags.image = args[i+1]
				i++
			}
		case "--pull":
			flags.pull = true
		case "--no-pull":
			flags.pull = false
		case "--verbose":
			flags.verbose = true
		case "--watch":
			flags.watch = true
		case "--wait":
			flags.wait = true
		case "--recreate", "--no-recreate":
			// accepted but ignored in Go CLI (compose handles it)
		default:
			positional = append(positional, args[i])
		}
	}
	return positional
}

// loadAndValidateConfig reads and validates langgraph.json. Returns the raw
// and validated config, or writes an error and returns nil.
func loadAndValidateConfig(configPath string, stderr io.Writer) (map[string]any, map[string]any, bool) {
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		errPrint(stderr, fmt.Sprintf("Path '%s' does not exist.", configPath))
		return nil, nil, false
	}

	raw, err := config.LoadRawConfigFile(configPath)
	if err != nil {
		errPrint(stderr, err.Error())
		return nil, nil, false
	}
	validated, err := config.ValidateConfigFile(configPath)
	if err != nil {
		errPrint(stderr, err.Error())
		return nil, nil, false
	}
	return raw, validated, true
}

// ---------------------------------------------------------------------------
// validate
// ---------------------------------------------------------------------------

func runValidate(args []string, stdout, stderr io.Writer) int {
	configPath := "langgraph.json"

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "-c", "--config":
			if i+1 < len(args) {
				configPath = args[i+1]
				i++
			} else {
				_, _ = fmt.Fprintln(stderr, "Error: --config requires a path argument")
				return 1
			}
		case "--help", "-h":
			_, _ = fmt.Fprintln(stdout, `Usage: langgraph validate [OPTIONS]

  Validate the LangGraph configuration file.

Options:
  -c, --config PATH  Path to configuration file (default: langgraph.json)
  --help             Show this message and exit.`)
			return 0
		default:
			_, _ = fmt.Fprintf(stderr, "Error: unexpected argument %q\n", args[i])
			return 1
		}
	}

	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		_, _ = fmt.Fprintf(stderr, "Error: Path '%s' does not exist.\n", configPath)
		return 1
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		_, _ = fmt.Fprintf(stderr, "%sError: %s%s\n", colorRed, err, colorReset)
		return 1
	}

	var rawAny any
	if err := json.Unmarshal(data, &rawAny); err != nil {
		_, _ = fmt.Fprintf(stderr, "Error: Invalid JSON in %s: %s\n", configPath, err.Error())
		return 1
	}
	rawConfig, ok := rawAny.(map[string]any)
	if !ok {
		_, _ = fmt.Fprintf(
			stderr,
			"%sError: Invalid config in %s: top-level JSON value must be an object.%s\n",
			colorRed,
			configPath,
			colorReset,
		)
		return 1
	}

	unknownWarnings := config.GetUnknownKeys(rawConfig)

	validated, validErr := config.ValidateConfigFile(configPath)
	if validErr != nil {
		_, _ = fmt.Fprintf(stderr, "%sError: %s%s\n", colorRed, validErr, colorReset)
		if len(unknownWarnings) > 0 {
			_, _ = fmt.Fprintln(stderr)
			for _, w := range unknownWarnings {
				_, _ = fmt.Fprintf(stderr, "  %swarning: %s%s\n", colorYellow, w, colorReset)
			}
		}
		return 1
	}

	graphs, _ := validated["graphs"].(map[string]any)
	numGraphs := len(graphs)
	plural := "s"
	if numGraphs == 1 {
		plural = ""
	}
	_, _ = fmt.Fprintf(stdout,
		"%sConfiguration file %s is valid. (%d graph%s found)%s\n",
		colorGreen, configPath, numGraphs, plural, colorReset)

	if len(unknownWarnings) > 0 {
		_, _ = fmt.Fprintln(stdout)
		for _, w := range unknownWarnings {
			_, _ = fmt.Fprintf(stdout, "  %swarning: %s%s\n", colorYellow, w, colorReset)
		}
	}

	return 0
}

// ---------------------------------------------------------------------------
// build
// ---------------------------------------------------------------------------

func runBuild(args []string, stdout, stderr io.Writer) int {
	for _, a := range args {
		if a == "--help" || a == "-h" {
			_, _ = fmt.Fprintln(stdout, `Usage: langgraph build [OPTIONS] [DOCKER_BUILD_ARGS]...

  Build LangGraph API server Docker image.

Options:
  -c, --config PATH           Path to configuration file (default: langgraph.json)
  -t, --tag TEXT              Tag for the docker image. [required]
  --pull / --no-pull          Pull latest images. (default: pull)
  --base-image TEXT           Base image for the LangGraph API server.
  --api-version TEXT          API server version for the base image.
  --engine-runtime-mode TEXT  Runtime mode (combined_queue_worker or distributed).
  --install-command TEXT      Custom install command.
  --build-command TEXT        Custom build command.
  --help                      Show this message and exit.`)
			return 0
		}
	}

	flags := newCommonFlags()
	passthrough := parseFlags(args, &flags)
	flags.passthrough = passthrough

	if flags.tag == "" {
		errPrint(stderr, "Missing option '--tag' / '-t'.")
		return 1
	}

	_, validated, ok := loadAndValidateConfig(flags.configPath, stderr)
	if !ok {
		return 1
	}

	baseImage := flags.baseImage
	if baseImage == "" {
		baseImage = config.DefaultBaseImage(validated, flags.engineRuntimeMode)
	}

	// Pull base image
	if flags.pull {
		tag := config.DockerTag(validated, baseImage, flags.apiVersion)
		_, _ = fmt.Fprintf(stdout, "Pulling %s...\n", tag)
		if err := lgexec.Run("docker", []string{"pull", tag}, lgexec.RunOpts{Verbose: true}); err != nil {
			_, _ = fmt.Fprintf(stderr, "%sWarning: failed to pull image: %s%s\n", colorYellow, err, colorReset)
		}
	}

	_, _ = fmt.Fprintln(stdout, "Building...")

	configJSON := deepCopyMap(validated)
	dockerfile, contexts, err := config.ConfigToDocker(flags.configPath, configJSON, config.DockerOpts{
		BaseImage:      baseImage,
		APIVersion:     flags.apiVersion,
		InstallCommand: flags.installCommand,
		BuildCommand:   flags.buildCommand,
	})
	if err != nil {
		errPrint(stderr, err.Error())
		return 1
	}

	buildArgs := []string{"build", "-f", "-", "-t", flags.tag}
	for k, v := range contexts {
		buildArgs = append(buildArgs, "--build-context", fmt.Sprintf("%s=%s", k, v))
	}
	buildArgs = append(buildArgs, flags.passthrough...)

	buildContext := filepath.Dir(absPath(flags.configPath))
	buildArgs = append(buildArgs, buildContext)

	if err := lgexec.Run("docker", buildArgs, lgexec.RunOpts{
		Stdin:   dockerfile,
		Verbose: true,
	}); err != nil {
		errPrint(stderr, fmt.Sprintf("Docker build failed: %s", err))
		return 1
	}

	_, _ = fmt.Fprintf(stdout, "%sSuccessfully built image: %s%s\n", colorGreen, flags.tag, colorReset)
	return 0
}

// ---------------------------------------------------------------------------
// dockerfile
// ---------------------------------------------------------------------------

func runDockerfile(args []string, stdout, stderr io.Writer) int {
	for _, a := range args {
		if a == "--help" || a == "-h" {
			_, _ = fmt.Fprintln(stdout, `Usage: langgraph dockerfile [OPTIONS] SAVE_PATH

  Generate a Dockerfile for the LangGraph API server.

Options:
  -c, --config PATH           Path to configuration file (default: langgraph.json)
  --base-image TEXT           Base image for the LangGraph API server.
  --api-version TEXT          API server version for the base image.
  --engine-runtime-mode TEXT  Runtime mode (combined_queue_worker or distributed).
  --add-docker-compose        Add docker-compose.yml and supporting files.
  --help                      Show this message and exit.`)
			return 0
		}
	}

	flags := newCommonFlags()
	addCompose := false
	var positional []string
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "-c", "--config":
			if i+1 < len(args) {
				flags.configPath = args[i+1]
				i++
			}
		case "--base-image":
			if i+1 < len(args) {
				flags.baseImage = args[i+1]
				i++
			}
		case "--api-version":
			if i+1 < len(args) {
				flags.apiVersion = args[i+1]
				i++
			}
		case "--engine-runtime-mode":
			if i+1 < len(args) {
				flags.engineRuntimeMode = args[i+1]
				i++
			}
		case "--add-docker-compose":
			addCompose = true
		default:
			positional = append(positional, args[i])
		}
	}

	if len(positional) < 1 {
		errPrint(stderr, "Missing argument 'SAVE_PATH'.")
		return 1
	}
	savePath := positional[0]

	_, validated, ok := loadAndValidateConfig(flags.configPath, stderr)
	if !ok {
		return 1
	}

	baseImage := flags.baseImage
	if baseImage == "" {
		baseImage = config.DefaultBaseImage(validated, flags.engineRuntimeMode)
	}

	configJSON := deepCopyMap(validated)
	dockerfile, _, err := config.ConfigToDocker(flags.configPath, configJSON, config.DockerOpts{
		BaseImage:  baseImage,
		APIVersion: flags.apiVersion,
	})
	if err != nil {
		errPrint(stderr, err.Error())
		return 1
	}

	if err := os.WriteFile(savePath, []byte(dockerfile), 0644); err != nil {
		errPrint(stderr, err.Error())
		return 1
	}
	_, _ = fmt.Fprintf(stdout, "%sDockerfile written to %s%s\n", colorGreen, savePath, colorReset)

	if addCompose {
		dir := filepath.Dir(savePath)
		composeStr, cerr := config.ConfigToCompose(flags.configPath, validated, config.ComposeOpts{
			BaseImage:         baseImage,
			APIVersion:        flags.apiVersion,
			EngineRuntimeMode: flags.engineRuntimeMode,
		})
		if cerr != nil {
			errPrint(stderr, cerr.Error())
			return 1
		}
		composePath := filepath.Join(dir, "docker-compose.yml")
		if err := os.WriteFile(composePath, []byte(composeStr), 0644); err != nil {
			errPrint(stderr, err.Error())
			return 1
		}
		_, _ = fmt.Fprintf(stdout, "%sDocker compose written to %s%s\n", colorGreen, composePath, colorReset)
	}

	return 0
}

// ---------------------------------------------------------------------------
// up
// ---------------------------------------------------------------------------

func runUp(args []string, stdout, stderr io.Writer) int {
	for _, a := range args {
		if a == "--help" || a == "-h" {
			_, _ = fmt.Fprintln(stdout, `Usage: langgraph up [OPTIONS]

  Launch LangGraph API server.

Options:
  -c, --config PATH           Path to configuration file (default: langgraph.json)
  -p, --port INTEGER          Port to expose (default: 8123)
  --pull / --no-pull          Pull latest images (default: pull)
  --recreate / --no-recreate  Recreate containers
  --verbose                   Show more output
  --watch                     Restart on file changes
  --wait                      Wait for services to start
  --postgres-uri TEXT         Postgres URI for database
  --debugger-port INTEGER     Port for the debugger UI
  --debugger-base-url TEXT    URL for debugger to access LangGraph API
  --base-image TEXT           Base image for the LangGraph API server
  --api-version TEXT          API server version for the base image
  --engine-runtime-mode TEXT  Runtime mode
  --image TEXT                Pre-built Docker image to use
  --help                      Show this message and exit.`)
			return 0
		}
	}

	flags := newCommonFlags()
	parseFlags(args, &flags)

	caps, err := docker.CheckCapabilities()
	if err != nil {
		errPrint(stderr, err.Error())
		return 1
	}

	_, validated, ok := loadAndValidateConfig(flags.configPath, stderr)
	if !ok {
		return 1
	}

	baseImage := flags.baseImage
	if baseImage == "" {
		baseImage = config.DefaultBaseImage(validated, flags.engineRuntimeMode)
	}

	// Generate compose YAML
	composeSnippet, cerr := config.ConfigToCompose(flags.configPath, validated, config.ComposeOpts{
		BaseImage:         baseImage,
		APIVersion:        flags.apiVersion,
		Image:             flags.image,
		Watch:             flags.watch,
		EngineRuntimeMode: flags.engineRuntimeMode,
	})
	if cerr != nil {
		errPrint(stderr, cerr.Error())
		return 1
	}

	infraYAML := docker.Compose(caps, docker.ComposeOpts{
		Port:              flags.port,
		DebuggerPort:      flags.debuggerPort,
		DebuggerBaseURL:   flags.debuggerBaseURL,
		PostgresURI:       flags.postgresURI,
		Image:             flags.image,
		EngineRuntimeMode: flags.engineRuntimeMode,
	})

	// Wrap the app config snippet into a valid compose overlay YAML.
	// ConfigToCompose returns content indented at the service-property level
	// (8 spaces), so we wrap it in the correct compose structure.
	appOverlayYAML := "services:\n    langgraph-api:" + composeSnippet

	// Pull
	if flags.pull && flags.image == "" {
		tag := config.DockerTag(validated, baseImage, flags.apiVersion)
		_, _ = fmt.Fprintf(stdout, "Pulling %s...\n", tag)
		_ = lgexec.Run("docker", []string{"pull", tag}, lgexec.RunOpts{Verbose: flags.verbose})
	}

	// Write infrastructure compose and app overlay to separate temp files.
	// Docker compose handles merging when given multiple -f flags.
	tmpInfra, err := os.CreateTemp("", "langgraph-infra-*.yml")
	if err != nil {
		errPrint(stderr, err.Error())
		return 1
	}
	defer os.Remove(tmpInfra.Name())
	_, _ = tmpInfra.WriteString(infraYAML)
	tmpInfra.Close()

	tmpApp, err := os.CreateTemp("", "langgraph-app-*.yml")
	if err != nil {
		errPrint(stderr, err.Error())
		return 1
	}
	defer os.Remove(tmpApp.Name())
	_, _ = tmpApp.WriteString(appOverlayYAML)
	tmpApp.Close()

	composeCmd := "docker"
	composeArgs := []string{"compose"}
	if caps.ComposeType == "standalone" {
		composeCmd = "docker-compose"
		composeArgs = nil
	}

	upArgs := append(composeArgs, "-f", tmpInfra.Name(), "-f", tmpApp.Name(), "up")
	if flags.wait {
		upArgs = append(upArgs, "--wait")
	} else {
		upArgs = append(upArgs, "-d")
	}

	_, _ = fmt.Fprintf(stdout, "%sStarting LangGraph API server...%s\n", colorCyan, colorReset)

	if err := lgexec.Run(composeCmd, upArgs, lgexec.RunOpts{
		Verbose: true,
		Dir:     filepath.Dir(absPath(flags.configPath)),
	}); err != nil {
		errPrint(stderr, fmt.Sprintf("docker compose up failed: %s", err))
		return 1
	}

	_, _ = fmt.Fprintf(stdout, "%sLangGraph API server is running at http://localhost:%d%s\n",
		colorGreen, flags.port, colorReset)
	return 0
}

// ---------------------------------------------------------------------------
// dev
// ---------------------------------------------------------------------------

func runDev(args []string, stdout, stderr io.Writer) int {
	for _, a := range args {
		if a == "--help" || a == "-h" {
			_, _ = fmt.Fprintln(stdout, `Usage: langgraph dev [OPTIONS]

  Run LangGraph API server in development mode with hot reloading.

Options:
  --host TEXT            Host to bind to (default: 127.0.0.1)
  --port INTEGER         Port to bind to (default: 2024)
  --config PATH          Path to configuration file (default: langgraph.json)
  --no-reload            Disable auto-reloading
  --no-browser           Skip opening the browser
  --debug-port INTEGER   Enable remote debugging on port
  --allow-blocking       Allow blocking I/O operations
  --tunnel               Expose via public tunnel
  --help                 Show this message and exit.`)
			return 0
		}
	}

	return runPythonCLI("dev", args, stdout, stderr)
}

// ---------------------------------------------------------------------------
// new
// ---------------------------------------------------------------------------

func runNew(args []string, stdout, stderr io.Writer) int {
	for _, a := range args {
		if a == "--help" || a == "-h" {
			_, _ = fmt.Fprintln(stdout, `Usage: langgraph new [OPTIONS] [PATH]

  Create a new LangGraph project from a template.

Options:
  --template TEXT  Template ID to use.
  --help           Show this message and exit.

Available templates:
`+templates.TemplateHelp())
			return 0
		}
	}

	var path, template string
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--template":
			if i+1 < len(args) {
				template = args[i+1]
				i++
			}
		default:
			if path == "" {
				path = args[i]
			}
		}
	}

	if path == "" {
		errPrint(stderr, "Missing argument 'PATH'. Usage: langgraph new [--template ID] PATH")
		return 1
	}
	if template == "" {
		errPrint(stderr, "Missing option '--template'. Available templates:\n"+templates.TemplateHelp())
		return 1
	}

	if err := templates.CreateNew(path, template); err != nil {
		errPrint(stderr, err.Error())
		return 1
	}

	_, _ = fmt.Fprintf(stdout, "%sCreated new LangGraph project at %s%s\n", colorGreen, path, colorReset)
	return 0
}

// ---------------------------------------------------------------------------
// deploy
// ---------------------------------------------------------------------------

func runDeploy(args []string, stdout, stderr io.Writer) int {
	return runPythonCLI("deploy", args, stdout, stderr)
}

func runPythonCLI(subcommand string, args []string, stdout, stderr io.Writer) int {
	pythonExe := os.Getenv("LANGGRAPH_CALLING_PYTHON")
	if pythonExe == "" {
		for _, name := range []string{"python3", "python"} {
			if p, _, err := lgexec.RunCollect("which", []string{name}); err == nil && strings.TrimSpace(p) != "" {
				pythonExe = strings.TrimSpace(p)
				break
			}
		}
	}
	if pythonExe == "" {
		errPrint(stderr, "Could not find Python interpreter. Set LANGGRAPH_CALLING_PYTHON.")
		return 1
	}

	pyArgs := append([]string{"-m", "langgraph_cli.cli", subcommand}, args...)
	if err := runPythonSubprocess(pythonExe, pyArgs, stdout, stderr); err != nil {
		if exitErr, ok := err.(*osexec.ExitError); ok {
			return exitErr.ExitCode()
		}
		errPrint(stderr, err.Error())
		return 1
	}
	return 0
}

func resolveDeployClient(args []string, stderr io.Writer) (apiKey, hostURL string, extra []string) {
	hostURL = deploy.DefaultHostURL
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--api-key":
			if i+1 < len(args) {
				apiKey = args[i+1]
				i++
			}
		case "--host-url":
			if i+1 < len(args) {
				hostURL = args[i+1]
				i++
			}
		default:
			extra = append(extra, args[i])
		}
	}
	if apiKey == "" {
		apiKey = deploy.ResolveAPIKey("", nil)
	}
	if apiKey == "" {
		errPrint(stderr, "API key required. Set --api-key or LANGSMITH_API_KEY environment variable.")
	}
	return
}

func runDeployMain(args []string, stdout, stderr io.Writer) int {
	for _, a := range args {
		if a == "--help" || a == "-h" {
			_, _ = fmt.Fprintln(stdout, `Usage: langgraph deploy [OPTIONS]

  Build and deploy a LangGraph image to LangSmith.

Options:
  --api-key TEXT             LangSmith API key
  --name TEXT                Deployment name
  --deployment-id TEXT       Existing deployment ID
  --deployment-type TEXT     dev or prod (default: dev)
  -c, --config PATH         Path to config (default: langgraph.json)
  --no-wait                  Skip waiting for deployment
  --verbose                  Show more output
  --remote / --no-remote     Force remote or local build
  -t, --tag TEXT             Image tag (default: latest)
  --base-image TEXT          Base image
  --help                     Show this message and exit.`)
			return 0
		}
	}

	// Parse deploy-specific flags
	var (
		apiKey, hostURL, name, deploymentID, deploymentType, tag string
		noWait, verbose, remote                                  bool
	)
	flags := newCommonFlags()
	deploymentType = "dev"
	tag = "latest"

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--api-key":
			if i+1 < len(args) {
				apiKey = args[i+1]
				i++
			}
		case "--host-url":
			if i+1 < len(args) {
				hostURL = args[i+1]
				i++
			}
		case "--name":
			if i+1 < len(args) {
				name = args[i+1]
				i++
			}
		case "--deployment-id":
			if i+1 < len(args) {
				deploymentID = args[i+1]
				i++
			}
		case "--deployment-type":
			if i+1 < len(args) {
				deploymentType = args[i+1]
				i++
			}
		case "-t", "--tag":
			if i+1 < len(args) {
				tag = args[i+1]
				i++
			}
		case "--no-wait":
			noWait = true
		case "--verbose":
			verbose = true
		case "--remote":
			remote = true
		default:
			// Parse common flags
			switch args[i] {
			case "-c", "--config":
				if i+1 < len(args) {
					flags.configPath = args[i+1]
					i++
				}
			case "--base-image":
				if i+1 < len(args) {
					flags.baseImage = args[i+1]
					i++
				}
			case "--api-version":
				if i+1 < len(args) {
					flags.apiVersion = args[i+1]
					i++
				}
			case "--pull", "--no-pull":
				// accept
			}
		}
	}

	if hostURL == "" {
		hostURL = deploy.DefaultHostURL
	}
	if apiKey == "" {
		apiKey = deploy.ResolveAPIKey("", nil)
	}
	if apiKey == "" {
		errPrint(stderr, "API key required. Set --api-key or LANGSMITH_API_KEY environment variable.")
		return 1
	}

	_, validated, ok := loadAndValidateConfig(flags.configPath, stderr)
	if !ok {
		return 1
	}

	_ = tag
	_ = deploymentType
	_ = verbose
	_ = remote
	_ = name
	_ = deploymentID
	_ = validated

	client := deploy.NewClient(hostURL, apiKey)

	// Resolve deployment name
	if name == "" && deploymentID == "" {
		// Default to current directory name
		absConfig, _ := filepath.Abs(flags.configPath)
		name = filepath.Base(filepath.Dir(absConfig))
	}

	// Find or create deployment
	var depID string
	if deploymentID != "" {
		depID = deploymentID
	} else {
		found, err := deploy.FindDeploymentIDByName(client, name)
		if err != nil {
			errPrint(stderr, err.Error())
			return 1
		}
		if found != "" {
			depID = found
			_, _ = fmt.Fprintf(stdout, "Found existing deployment: %s\n", depID)
		} else {
			_, _ = fmt.Fprintf(stdout, "Creating deployment '%s'...\n", name)
			resp, err := client.CreateDeployment(name, deploymentType, "internal_docker", "", nil)
			if err != nil {
				errPrint(stderr, err.Error())
				return 1
			}
			depID, _ = resp["id"].(string)
			_, _ = fmt.Fprintf(stdout, "Created deployment: %s\n", depID)
		}
	}

	// Build locally
	baseImage := flags.baseImage
	if baseImage == "" {
		baseImage = config.DefaultBaseImage(validated, "combined_queue_worker")
	}
	imgTag := fmt.Sprintf("langgraph-%s:%s", deploy.NormalizeImageName(name), tag)

	_, _ = fmt.Fprintln(stdout, "Building image...")
	buildConfig := deepCopyMap(validated)
	dockerfile, contexts, err := config.ConfigToDocker(flags.configPath, buildConfig, config.DockerOpts{
		BaseImage:  baseImage,
		APIVersion: flags.apiVersion,
	})
	if err != nil {
		errPrint(stderr, err.Error())
		return 1
	}

	buildArgs := []string{"build", "-f", "-", "-t", imgTag}
	for k, v := range contexts {
		buildArgs = append(buildArgs, "--build-context", fmt.Sprintf("%s=%s", k, v))
	}
	buildArgs = append(buildArgs, filepath.Dir(absPath(flags.configPath)))

	if err := lgexec.Run("docker", buildArgs, lgexec.RunOpts{Stdin: dockerfile, Verbose: verbose}); err != nil {
		errPrint(stderr, fmt.Sprintf("Build failed: %s", err))
		return 1
	}

	// Push
	_, _ = fmt.Fprintln(stdout, "Requesting push token...")
	tokenResp, err := client.RequestPushToken(depID)
	if err != nil {
		errPrint(stderr, err.Error())
		return 1
	}
	registryURL, _ := tokenResp["registry_url"].(string)
	token, _ := tokenResp["token"].(string)

	remoteTag := fmt.Sprintf("%s:%s", registryURL, tag)
	_ = lgexec.Run("docker", []string{"tag", imgTag, remoteTag}, lgexec.RunOpts{})
	_ = lgexec.Run("docker", []string{"login", "-u", "oauth2accesstoken", "-p", token, registryURL}, lgexec.RunOpts{})

	_, _ = fmt.Fprintln(stdout, "Pushing image...")
	if err := lgexec.Run("docker", []string{"push", remoteTag}, lgexec.RunOpts{Verbose: verbose}); err != nil {
		errPrint(stderr, fmt.Sprintf("Push failed: %s", err))
		return 1
	}

	// Update deployment
	_, _ = fmt.Fprintln(stdout, "Updating deployment...")
	envVars := deploy.ParseEnvFromConfig(validated, flags.configPath)
	secrets := deploy.SecretsFromEnv(envVars)

	_, err = client.UpdateDeployment(depID, remoteTag, secrets)
	if err != nil {
		errPrint(stderr, err.Error())
		return 1
	}

	if !noWait {
		_, _ = fmt.Fprintln(stdout, "Waiting for deployment...")
		finalStatus, pollErr := deploy.PollDeploymentStatus(client, depID, 300, 2, func(status string) {
			_, _ = fmt.Fprintf(stdout, "  Status: %s\n", status)
		})
		if pollErr != nil {
			errPrint(stderr, pollErr.Error())
			return 1
		}
		if finalStatus != "DEPLOYED" {
			errPrint(stderr, fmt.Sprintf("Deployment failed with status: %s", finalStatus))
			return 1
		}
	}

	_, _ = fmt.Fprintf(stdout, "%sDeployment updated successfully!%s\n", colorGreen, colorReset)
	return 0
}

func runDeployList(args []string, stdout, stderr io.Writer) int {
	for _, a := range args {
		if a == "--help" || a == "-h" {
			_, _ = fmt.Fprintln(stdout, `Usage: langgraph deploy list [OPTIONS]

  List LangSmith Deployments.

Options:
  --api-key TEXT          API key
  --name-contains TEXT    Filter by name
  --help                  Show this message and exit.`)
			return 0
		}
	}

	var nameContains string
	apiKey, hostURL, extra := resolveDeployClient(args, stderr)
	if apiKey == "" {
		return 1
	}
	for i := 0; i < len(extra); i++ {
		if extra[i] == "--name-contains" && i+1 < len(extra) {
			nameContains = extra[i+1]
			i++
		}
	}

	client := deploy.NewClient(hostURL, apiKey)
	resp, err := client.ListDeployments(nameContains)
	if err != nil {
		errPrint(stderr, err.Error())
		return 1
	}

	deployments, _ := resp["deployments"].([]any)
	if len(deployments) == 0 {
		_, _ = fmt.Fprintln(stdout, "No deployments found.")
		return 0
	}

	// Format table
	_, _ = fmt.Fprintf(stdout, "%-38s  %-30s  %s\n", "Deployment ID", "Name", "URL")
	_, _ = fmt.Fprintf(stdout, "%-38s  %-30s  %s\n", strings.Repeat("-", 38), strings.Repeat("-", 30), strings.Repeat("-", 40))
	for _, d := range deployments {
		dep, _ := d.(map[string]any)
		id, _ := dep["id"].(string)
		name, _ := dep["name"].(string)
		url := "-"
		if sc, ok := dep["source_config"].(map[string]any); ok {
			if u, ok := sc["custom_url"].(string); ok && u != "" {
				url = u
			}
		}
		_, _ = fmt.Fprintf(stdout, "%-38s  %-30s  %s\n", id, name, url)
	}
	return 0
}

func runDeployRevisions(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		_, _ = fmt.Fprintln(stdout, `Usage: langgraph deploy revisions COMMAND [ARGS]...

Commands:
  list  List revisions for a deployment.`)
		return 0
	}
	switch args[0] {
	case "list":
		return runDeployRevisionsList(args[1:], stdout, stderr)
	case "--help", "-h":
		_, _ = fmt.Fprintln(stdout, `Usage: langgraph deploy revisions COMMAND [ARGS]...

Commands:
  list  List revisions for a deployment.`)
		return 0
	default:
		errPrint(stderr, fmt.Sprintf("Unknown revisions command: %s", args[0]))
		return 1
	}
}

func runDeployRevisionsList(args []string, stdout, stderr io.Writer) int {
	for _, a := range args {
		if a == "--help" || a == "-h" {
			_, _ = fmt.Fprintln(stdout, `Usage: langgraph deploy revisions list [OPTIONS] DEPLOYMENT_ID

  List revisions for a LangSmith Deployment.

Options:
  --api-key TEXT     API key
  --limit INTEGER    Max revisions (default: 10)
  --help             Show this message and exit.`)
			return 0
		}
	}

	limit := 10
	apiKey, hostURL, extra := resolveDeployClient(args, stderr)
	if apiKey == "" {
		return 1
	}

	var deploymentID string
	for i := 0; i < len(extra); i++ {
		if extra[i] == "--limit" && i+1 < len(extra) {
			if n, err := strconv.Atoi(extra[i+1]); err == nil {
				limit = n
			}
			i++
		} else if !strings.HasPrefix(extra[i], "-") && deploymentID == "" {
			deploymentID = extra[i]
		}
	}

	if deploymentID == "" {
		errPrint(stderr, "Missing argument 'DEPLOYMENT_ID'.")
		return 1
	}

	client := deploy.NewClient(hostURL, apiKey)
	resp, err := client.ListRevisions(deploymentID, limit)
	if err != nil {
		errPrint(stderr, err.Error())
		return 1
	}

	revisions, _ := resp["revisions"].([]any)
	if len(revisions) == 0 {
		_, _ = fmt.Fprintln(stdout, "No revisions found.")
		return 0
	}

	_, _ = fmt.Fprintf(stdout, "%-38s  %-15s  %s\n", "Revision ID", "Status", "Created At")
	_, _ = fmt.Fprintf(stdout, "%-38s  %-15s  %s\n", strings.Repeat("-", 38), strings.Repeat("-", 15), strings.Repeat("-", 25))
	for _, r := range revisions {
		rev, _ := r.(map[string]any)
		id, _ := rev["id"].(string)
		status, _ := rev["status"].(string)
		created, _ := rev["created_at"].(string)
		_, _ = fmt.Fprintf(stdout, "%-38s  %-15s  %s\n", id, status, created)
	}
	return 0
}

func runDeployDelete(args []string, stdout, stderr io.Writer) int {
	for _, a := range args {
		if a == "--help" || a == "-h" {
			_, _ = fmt.Fprintln(stdout, `Usage: langgraph deploy delete [OPTIONS] DEPLOYMENT_ID

  Delete a LangSmith Deployment.

Options:
  --api-key TEXT   API key
  --force          Delete without confirmation
  --help           Show this message and exit.`)
			return 0
		}
	}

	force := false
	apiKey, hostURL, extra := resolveDeployClient(args, stderr)
	if apiKey == "" {
		return 1
	}

	var deploymentID string
	for i := 0; i < len(extra); i++ {
		if extra[i] == "--force" {
			force = true
		} else if !strings.HasPrefix(extra[i], "-") && deploymentID == "" {
			deploymentID = extra[i]
		}
	}

	if deploymentID == "" {
		errPrint(stderr, "Missing argument 'DEPLOYMENT_ID'.")
		return 1
	}

	if !force {
		_, _ = fmt.Fprintf(stdout, "Are you sure you want to delete deployment %s? [y/N] ", deploymentID)
		var answer string
		_, _ = fmt.Fscanln(os.Stdin, &answer)
		if answer != "y" && answer != "Y" {
			_, _ = fmt.Fprintln(stdout, "Aborted.")
			return 0
		}
	}

	client := deploy.NewClient(hostURL, apiKey)
	if err := client.DeleteDeployment(deploymentID); err != nil {
		errPrint(stderr, err.Error())
		return 1
	}

	_, _ = fmt.Fprintf(stdout, "%sDeployment %s deleted.%s\n", colorGreen, deploymentID, colorReset)
	return 0
}

func runDeployLogs(args []string, stdout, stderr io.Writer) int {
	for _, a := range args {
		if a == "--help" || a == "-h" {
			_, _ = fmt.Fprintln(stdout, `Usage: langgraph deploy logs [OPTIONS]

  Fetch LangSmith Deployment logs.

Options:
  --api-key TEXT           API key
  --name TEXT              Deployment name
  --deployment-id TEXT     Deployment ID
  --type TEXT              Log type: deploy or build (default: deploy)
  --revision-id TEXT       Specific revision ID
  --level TEXT             Filter by log level
  --limit INTEGER          Max entries (default: 100)
  --query TEXT             Search string
  --follow                 Continuously poll for new logs
  --help                   Show this message and exit.`)
			return 0
		}
	}

	var (
		name, deploymentID, logType, revisionID, level, query string
		limit                                                 int
		follow                                                bool
	)
	logType = "deploy"
	limit = 100

	apiKey, hostURL, extra := resolveDeployClient(args, stderr)
	if apiKey == "" {
		return 1
	}

	for i := 0; i < len(extra); i++ {
		switch extra[i] {
		case "--name":
			if i+1 < len(extra) {
				name = extra[i+1]
				i++
			}
		case "--deployment-id":
			if i+1 < len(extra) {
				deploymentID = extra[i+1]
				i++
			}
		case "--type":
			if i+1 < len(extra) {
				logType = extra[i+1]
				i++
			}
		case "--revision-id":
			if i+1 < len(extra) {
				revisionID = extra[i+1]
				i++
			}
		case "--level":
			if i+1 < len(extra) {
				level = extra[i+1]
				i++
			}
		case "--limit":
			if i+1 < len(extra) {
				if n, err := strconv.Atoi(extra[i+1]); err == nil {
					limit = n
				}
				i++
			}
		case "--query", "-q":
			if i+1 < len(extra) {
				query = extra[i+1]
				i++
			}
		case "--follow", "-f":
			follow = true
		}
	}

	if deploymentID == "" && name == "" {
		errPrint(stderr, "Provide --deployment-id or --name.")
		return 1
	}

	client := deploy.NewClient(hostURL, apiKey)

	if deploymentID == "" {
		found, err := deploy.FindDeploymentIDByName(client, name)
		if err != nil {
			errPrint(stderr, err.Error())
			return 1
		}
		if found == "" {
			errPrint(stderr, fmt.Sprintf("No deployment found with name '%s'.", name))
			return 1
		}
		deploymentID = found
	}

	payload := map[string]any{
		"limit": limit,
		"order": "desc",
	}
	if level != "" {
		payload["level"] = level
	}
	if query != "" {
		payload["query"] = query
	}

	if follow {
		payload["order"] = "asc"
		seen := make(map[string]bool)
		for {
			var resp map[string]any
			var err error
			if logType == "build" {
				if revisionID == "" {
					revResp, rerr := client.ListRevisions(deploymentID, 1)
					if rerr != nil {
						errPrint(stderr, rerr.Error())
						return 1
					}
					revisions, _ := revResp["revisions"].([]any)
					if len(revisions) > 0 {
						rev, _ := revisions[0].(map[string]any)
						revisionID, _ = rev["id"].(string)
					}
				}
				if revisionID == "" {
					time.Sleep(2 * time.Second)
					continue
				}
				resp, err = client.GetBuildLogs(deploymentID, revisionID, payload)
			} else {
				resp, err = client.GetDeployLogs(deploymentID, payload, revisionID)
			}
			if err != nil {
				errPrint(stderr, err.Error())
				return 1
			}
			logs, _ := resp["logs"].([]any)
			for _, l := range logs {
				entry, _ := l.(map[string]any)
				id, _ := entry["id"].(string)
				if id == "" {
					// Fall back to timestamp+message as key
					ts, _ := entry["timestamp"].(string)
					msg, _ := entry["message"].(string)
					id = ts + "|" + msg
				}
				if seen[id] {
					continue
				}
				seen[id] = true
				ts, _ := entry["timestamp"].(string)
				lvl, _ := entry["level"].(string)
				msg, _ := entry["message"].(string)
				if ts != "" {
					_, _ = fmt.Fprintf(stdout, "[%s] ", ts)
				}
				if lvl != "" {
					_, _ = fmt.Fprintf(stdout, "[%s] ", lvl)
				}
				_, _ = fmt.Fprintln(stdout, msg)
			}
			time.Sleep(2 * time.Second)
		}
	}

	var resp map[string]any
	var err error
	if logType == "build" {
		if revisionID == "" {
			// Get latest revision
			revResp, rerr := client.ListRevisions(deploymentID, 1)
			if rerr != nil {
				errPrint(stderr, rerr.Error())
				return 1
			}
			revisions, _ := revResp["revisions"].([]any)
			if len(revisions) > 0 {
				rev, _ := revisions[0].(map[string]any)
				revisionID, _ = rev["id"].(string)
			}
		}
		if revisionID == "" {
			errPrint(stderr, "No revisions found for build logs.")
			return 1
		}
		resp, err = client.GetBuildLogs(deploymentID, revisionID, payload)
	} else {
		resp, err = client.GetDeployLogs(deploymentID, payload, revisionID)
	}

	if err != nil {
		errPrint(stderr, err.Error())
		return 1
	}

	logs, _ := resp["logs"].([]any)
	for i := len(logs) - 1; i >= 0; i-- {
		entry, _ := logs[i].(map[string]any)
		ts, _ := entry["timestamp"].(string)
		lvl, _ := entry["level"].(string)
		msg, _ := entry["message"].(string)
		if ts != "" {
			_, _ = fmt.Fprintf(stdout, "[%s] ", ts)
		}
		if lvl != "" {
			_, _ = fmt.Fprintf(stdout, "[%s] ", lvl)
		}
		_, _ = fmt.Fprintln(stdout, msg)
	}
	return 0
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func absPath(p string) string {
	abs, err := filepath.Abs(p)
	if err != nil {
		return p
	}
	return abs
}

func deepCopyMap(m map[string]any) map[string]any {
	data, _ := json.Marshal(m)
	var out map[string]any
	_ = json.Unmarshal(data, &out)
	return out
}
