// Package docker provides Docker compose generation, capability detection,
// and image building for the LangGraph CLI.
package docker

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/langchain-ai/langgraph/libs/cli/internal/config"
)

// DefaultPostgresURI is the default connection string used when no custom
// Postgres URI is provided.
const DefaultPostgresURI = "postgres://postgres:postgres@langgraph-postgres:5432/postgres?sslmode=disable"

// Version represents a semantic version with major, minor, and patch components.
type Version struct {
	Major, Minor, Patch int
}

// GreaterOrEqual returns true if v >= other.
func (v Version) GreaterOrEqual(other Version) bool {
	if v.Major != other.Major {
		return v.Major > other.Major
	}
	if v.Minor != other.Minor {
		return v.Minor > other.Minor
	}
	return v.Patch >= other.Patch
}

// DockerCapabilities describes the Docker environment available on the host.
type DockerCapabilities struct {
	VersionDocker            Version
	VersionCompose           Version
	HealthcheckStartInterval bool
	ComposeType              string // "plugin" or "standalone"
}

// ComposeOpts configures the generated docker-compose YAML.
type ComposeOpts struct {
	Port              int
	DebuggerPort      int    // 0 means no debugger
	DebuggerBaseURL   string // optional base URL for the debugger
	PostgresURI       string // empty means use DefaultPostgresURI
	Image             string // pre-built image name
	BaseImage         string
	APIVersion        string
	EngineRuntimeMode string // "combined_queue_worker" or "distributed"
}

// BuildImageOpts configures docker image building.
type BuildImageOpts struct {
	ConfigPath     string
	ConfigJSON     map[string]any
	BaseImage      string
	APIVersion     string
	Pull           bool
	Tag            string
	Passthrough    []string
	InstallCommand string
	BuildCommand   string
	DockerCommand  []string // default: ["docker", "build"]
	ExtraFlags     []string
	Verbose        bool
}

// OrderedMap preserves insertion order for map keys.
type OrderedMap struct {
	Keys   []string
	Values map[string]any
}

// NewOrderedMap creates an empty OrderedMap.
func NewOrderedMap() *OrderedMap {
	return &OrderedMap{
		Values: make(map[string]any),
	}
}

// Set adds or updates a key-value pair, preserving insertion order.
func (om *OrderedMap) Set(key string, value any) {
	if _, exists := om.Values[key]; !exists {
		om.Keys = append(om.Keys, key)
	}
	om.Values[key] = value
}

// Get retrieves the value for a key.
func (om *OrderedMap) Get(key string) (any, bool) {
	v, ok := om.Values[key]
	return v, ok
}

// ---------------------------------------------------------------------------
// ParseVersion
// ---------------------------------------------------------------------------

// ParseVersion parses a version string like "1.2.3", "v1.2.3-alpha+build",
// "1.2", or "1" into a Version.
func ParseVersion(version string) Version {
	parts := strings.SplitN(version, ".", 3)

	major := "0"
	minor := "0"
	patch := "0"

	switch len(parts) {
	case 1:
		major = parts[0]
	case 2:
		major = parts[0]
		minor = parts[1]
	default:
		major = parts[0]
		minor = parts[1]
		patch = parts[2]
	}

	// Strip "v" prefix from major
	major = strings.TrimPrefix(major, "v")

	// Strip "-" and "+" suffixes from patch
	if idx := strings.IndexAny(patch, "-+"); idx >= 0 {
		patch = patch[:idx]
	}

	majorInt, _ := strconv.Atoi(major)
	minorInt, _ := strconv.Atoi(minor)
	patchInt, _ := strconv.Atoi(patch)

	return Version{Major: majorInt, Minor: minorInt, Patch: patchInt}
}

// ---------------------------------------------------------------------------
// CanBuildLocally
// ---------------------------------------------------------------------------

// CanBuildLocally checks whether local deployment builds can run on this machine.
// It returns (ok, errorMessage). If ok is true, errorMessage is empty.
func CanBuildLocally() (bool, string) {
	if _, err := exec.LookPath("docker"); err != nil {
		return false, "Docker is required but not installed.\n" +
			"Install Docker Desktop: https://docs.docker.com/get-docker/"
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "docker", "info")
	cmd.Stdout = nil
	cmd.Stderr = nil
	if err := cmd.Run(); err != nil {
		return false, "Docker is installed but not running.\nStart Docker and try again."
	}

	if runtime.GOARCH != "amd64" {
		ctx2, cancel2 := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel2()

		buildx := exec.CommandContext(ctx2, "docker", "buildx", "version")
		buildx.Stdout = nil
		buildx.Stderr = nil
		if err := buildx.Run(); err != nil {
			arch := runtime.GOARCH
			// Try to match Python's platform.machine() naming for the error message
			if arch == "arm64" {
				arch = "aarch64"
			}
			return false, "Docker Buildx is required but not installed.\n" +
				"Your machine architecture (" + arch + ") requires Buildx to cross-compile images for linux/amd64.\n" +
				"Install Buildx: https://docs.docker.com/build/install-buildx/"
		}
	}
	return true, ""
}

// ---------------------------------------------------------------------------
// CheckCapabilities
// ---------------------------------------------------------------------------

// CheckCapabilities detects the Docker and Docker Compose versions available
// on the host and returns a DockerCapabilities describing them.
func CheckCapabilities() (*DockerCapabilities, error) {
	if _, err := exec.LookPath("docker"); err != nil {
		return nil, fmt.Errorf("Docker not installed")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	out, err := exec.CommandContext(ctx, "docker", "info", "-f", "{{json .}}").Output()
	if err != nil {
		return nil, fmt.Errorf("Docker not installed or not running")
	}

	var info map[string]any
	if err := json.Unmarshal(out, &info); err != nil {
		return nil, fmt.Errorf("Docker not installed or not running")
	}

	serverVersion, _ := info["ServerVersion"].(string)
	if serverVersion == "" {
		return nil, fmt.Errorf("Docker not running")
	}

	// Try to find compose plugin
	var composeVersionStr string
	composeType := "plugin"

	found := false
	if clientInfo, ok := info["ClientInfo"].(map[string]any); ok {
		if plugins, ok := clientInfo["Plugins"].([]any); ok {
			for _, p := range plugins {
				pm, ok := p.(map[string]any)
				if !ok {
					continue
				}
				name, _ := pm["Name"].(string)
				if name == "compose" {
					composeVersionStr, _ = pm["Version"].(string)
					found = true
					break
				}
			}
		}
	}

	if !found {
		// Fall back to standalone docker-compose
		if _, err := exec.LookPath("docker-compose"); err != nil {
			return nil, fmt.Errorf("Docker Compose not installed")
		}

		ctx2, cancel2 := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel2()

		out2, err := exec.CommandContext(ctx2, "docker-compose", "--version", "--short").Output()
		if err != nil {
			return nil, fmt.Errorf("Docker Compose not installed")
		}
		composeVersionStr = strings.TrimSpace(string(out2))
		composeType = "standalone"
	}

	dockerVersion := ParseVersion(serverVersion)
	composeVersion := ParseVersion(composeVersionStr)

	return &DockerCapabilities{
		VersionDocker:            dockerVersion,
		VersionCompose:           composeVersion,
		HealthcheckStartInterval: dockerVersion.GreaterOrEqual(Version{25, 0, 0}),
		ComposeType:              composeType,
	}, nil
}

// ---------------------------------------------------------------------------
// DebuggerCompose
// ---------------------------------------------------------------------------

// DebuggerCompose returns a service config map for the langgraph-debugger
// container, or nil if port is 0 (no debugger requested).
func DebuggerCompose(port int, baseURL string) *OrderedMap {
	if port == 0 {
		return nil
	}

	dependsOn := NewOrderedMap()
	postgresCondition := NewOrderedMap()
	postgresCondition.Set("condition", "service_healthy")
	dependsOn.Set("langgraph-postgres", postgresCondition)

	service := NewOrderedMap()
	service.Set("image", "langchain/langgraph-debugger")
	service.Set("restart", "on-failure")
	service.Set("depends_on", dependsOn)
	service.Set("ports", []any{fmt.Sprintf(`"%d:3968"`, port)})

	if baseURL != "" {
		env := NewOrderedMap()
		env.Set("VITE_STUDIO_LOCAL_GRAPH_URL", baseURL)
		service.Set("environment", env)
	}

	result := NewOrderedMap()
	result.Set("langgraph-debugger", service)
	return result
}

// ---------------------------------------------------------------------------
// DictToYAML
// ---------------------------------------------------------------------------

// DictToYAML converts an OrderedMap to a YAML string. For top-level keys
// (indent < 2) it adds a blank line between entries (except the first).
func DictToYAML(d *OrderedMap, indent int) string {
	var b strings.Builder
	for idx, key := range d.Keys {
		// Add blank line between top-level entries (except first)
		if idx >= 1 && indent < 2 {
			b.WriteString("\n")
		}

		space := strings.Repeat("    ", indent)
		value := d.Values[key]

		switch v := value.(type) {
		case *OrderedMap:
			b.WriteString(fmt.Sprintf("%s%s:\n", space, key))
			b.WriteString(DictToYAML(v, indent+1))
		case []any:
			b.WriteString(fmt.Sprintf("%s%s:\n", space, key))
			for _, item := range v {
				b.WriteString(fmt.Sprintf("%s    - %v\n", space, item))
			}
		default:
			b.WriteString(fmt.Sprintf("%s%s: %v\n", space, key, value))
		}
	}
	return b.String()
}

// ---------------------------------------------------------------------------
// ComposeAsDict
// ---------------------------------------------------------------------------

// ComposeAsDict builds the docker-compose configuration as an OrderedMap.
func ComposeAsDict(caps *DockerCapabilities, opts ComposeOpts) *OrderedMap {
	postgresURI := opts.PostgresURI
	includeDB := false
	if postgresURI == "" {
		includeDB = true
		postgresURI = DefaultPostgresURI
	}

	services := NewOrderedMap()

	// --- Redis service ---
	redisHealthcheck := NewOrderedMap()
	redisHealthcheck.Set("test", "redis-cli ping")
	redisHealthcheck.Set("interval", "5s")
	redisHealthcheck.Set("timeout", "1s")
	redisHealthcheck.Set("retries", 5)

	redisService := NewOrderedMap()
	redisService.Set("image", "redis:6")
	redisService.Set("healthcheck", redisHealthcheck)
	services.Set("langgraph-redis", redisService)

	// --- Postgres service (if no custom URI) ---
	if includeDB {
		pgEnv := NewOrderedMap()
		pgEnv.Set("POSTGRES_DB", "postgres")
		pgEnv.Set("POSTGRES_USER", "postgres")
		pgEnv.Set("POSTGRES_PASSWORD", "postgres")

		pgHealthcheck := NewOrderedMap()
		pgHealthcheck.Set("test", "pg_isready -U postgres")
		pgHealthcheck.Set("start_period", "10s")
		pgHealthcheck.Set("timeout", "1s")
		pgHealthcheck.Set("retries", 5)

		if caps.HealthcheckStartInterval {
			pgHealthcheck.Set("interval", "60s")
			pgHealthcheck.Set("start_interval", "1s")
		} else {
			pgHealthcheck.Set("interval", "5s")
		}

		pgService := NewOrderedMap()
		pgService.Set("image", "pgvector/pgvector:pg16")
		pgService.Set("ports", []any{`"5433:5432"`})
		pgService.Set("environment", pgEnv)
		pgService.Set("command", []any{"postgres", "-c", "shared_preload_libraries=vector"})
		pgService.Set("volumes", []any{"langgraph-data:/var/lib/postgresql/data"})
		pgService.Set("healthcheck", pgHealthcheck)

		services.Set("langgraph-postgres", pgService)
	}

	// --- Debugger service (optional) ---
	if opts.DebuggerPort != 0 {
		debuggerMap := DebuggerCompose(opts.DebuggerPort, opts.DebuggerBaseURL)
		if debuggerMap != nil {
			debuggerService, _ := debuggerMap.Get("langgraph-debugger")
			services.Set("langgraph-debugger", debuggerService)
		}
	}

	// --- langgraph-api service ---
	apiEnv := NewOrderedMap()
	apiEnv.Set("REDIS_URI", "redis://langgraph-redis:6379")
	apiEnv.Set("POSTGRES_URI", postgresURI)

	if opts.EngineRuntimeMode == "distributed" {
		apiEnv.Set("N_JOBS_PER_WORKER", `"0"`)
	}

	apiDependsOn := NewOrderedMap()
	redisCondition := NewOrderedMap()
	redisCondition.Set("condition", "service_healthy")
	apiDependsOn.Set("langgraph-redis", redisCondition)

	if includeDB {
		pgCondition := NewOrderedMap()
		pgCondition.Set("condition", "service_healthy")
		apiDependsOn.Set("langgraph-postgres", pgCondition)
	}

	apiService := NewOrderedMap()
	apiService.Set("ports", []any{fmt.Sprintf(`"%d:8000"`, opts.Port)})
	apiService.Set("depends_on", apiDependsOn)
	apiService.Set("environment", apiEnv)

	if opts.Image != "" {
		apiService.Set("image", opts.Image)
	}

	if caps.HealthcheckStartInterval {
		apiHealthcheck := NewOrderedMap()
		apiHealthcheck.Set("test", "python /api/healthcheck.py")
		apiHealthcheck.Set("interval", "60s")
		apiHealthcheck.Set("start_interval", "1s")
		apiHealthcheck.Set("start_period", "10s")
		apiService.Set("healthcheck", apiHealthcheck)
	}

	services.Set("langgraph-api", apiService)

	// --- Build final compose dict ---
	composeDict := NewOrderedMap()
	if includeDB {
		volumes := NewOrderedMap()
		volumeDriver := NewOrderedMap()
		volumeDriver.Set("driver", "local")
		volumes.Set("langgraph-data", volumeDriver)
		composeDict.Set("volumes", volumes)
	}
	composeDict.Set("services", services)

	return composeDict
}

// ---------------------------------------------------------------------------
// Compose
// ---------------------------------------------------------------------------

// Compose generates a docker-compose YAML string from the given capabilities
// and options.
func Compose(caps *DockerCapabilities, opts ComposeOpts) string {
	d := ComposeAsDict(caps, opts)
	return DictToYAML(d, 0)
}

// ---------------------------------------------------------------------------
// BuildDockerImage
// ---------------------------------------------------------------------------

// BuildDockerImage builds a Docker image from a LangGraph configuration.
// It shells out to docker build (or a custom docker command) with the
// generated Dockerfile piped via stdin.
func BuildDockerImage(opts BuildImageOpts) error {
	dockerCmd := opts.DockerCommand
	if len(dockerCmd) == 0 {
		dockerCmd = []string{"docker", "build"}
	}

	// Pull the base image first if requested.
	if opts.Pull {
		pullCmd := exec.Command("docker", "pull", opts.Tag)
		pullCmd.Stdout = os.Stdout
		pullCmd.Stderr = os.Stderr
		if err := pullCmd.Run(); err != nil {
			return fmt.Errorf("failed to pull image %s: %w", opts.Tag, err)
		}
	}

	// Build the docker build arguments.
	args := []string{
		"-f", "-", // read Dockerfile from stdin
		"-t", opts.Tag,
	}

	// Determine build context.
	buildContext := "."
	if opts.ConfigPath != "" {
		// Use the parent directory of the config file by default.
		idx := strings.LastIndex(opts.ConfigPath, "/")
		if idx >= 0 {
			buildContext = opts.ConfigPath[:idx]
		}
	}

	// Generate the Dockerfile using the real config.ConfigToDocker.
	dockerfile, additionalContexts, err := config.ConfigToDocker(opts.ConfigPath, opts.ConfigJSON, config.DockerOpts{
		BaseImage:      opts.BaseImage,
		APIVersion:     opts.APIVersion,
		InstallCommand: opts.InstallCommand,
		BuildCommand:   opts.BuildCommand,
	})
	if err != nil {
		return fmt.Errorf("generating Dockerfile: %w", err)
	}

	// Add additional build contexts (for dependencies outside the main context).
	for name, path := range additionalContexts {
		args = append(args, "--build-context", fmt.Sprintf("%s=%s", name, path))
	}

	// Assemble the full command.
	fullArgs := append(dockerCmd[1:], args...)
	fullArgs = append(fullArgs, opts.ExtraFlags...)
	fullArgs = append(fullArgs, opts.Passthrough...)
	fullArgs = append(fullArgs, buildContext)

	cmd := exec.Command(dockerCmd[0], fullArgs...)
	cmd.Stdin = strings.NewReader(dockerfile)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	return cmd.Run()
}
